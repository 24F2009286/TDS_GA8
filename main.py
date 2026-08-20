"""
Immutable, Leakage-Safe Training Corpus builder.

POST /build-corpus

See the assignment spec for exact behavior. Key design decisions /
assumptions made where the spec is ambiguous are marked with `# ASSUMPTION:`.
"""

import base64
import hashlib
import json
import math
import re
import struct
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)


def parse_timestamp(s: str):
    """Return an aware UTC datetime, or None if invalid per spec."""
    if not isinstance(s, str):
        return None
    m = TS_RE.match(s)
    if not m:
        return None
    year, month, day, hour, minute, second, frac, tz = m.groups()
    year, month, day = int(year), int(month), int(day)
    hour, minute, second = int(hour), int(minute), int(second)

    if not (1 <= month <= 12):
        return None
    # Validate calendar day (handles leap years correctly via datetime).
    try:
        naive = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None

    ms = 0
    if frac:
        ms = int(frac.ljust(3, "0")[:3])

    if tz == "Z":
        offset_minutes = 0
    else:
        sign = 1 if tz[0] == "+" else -1
        oh, om = int(tz[1:3]), int(tz[4:6])
        if oh > 14 or om > 59:
            return None
        if oh == 14 and om != 0:
            return None
        offset_minutes = sign * (oh * 60 + om)

    naive = naive.replace(microsecond=ms * 1000)
    utc_dt = naive - timedelta(minutes=offset_minutes)
    return utc_dt.replace(tzinfo=timezone.utc)


def format_utc(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


# ---------------------------------------------------------------------------
# CRC32C (Castagnoli) - not the same as zlib.crc32
# ---------------------------------------------------------------------------

_CRC32C_TABLE = None


def _build_crc32c_table():
    global _CRC32C_TABLE
    poly = 0x82F63B78
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (poly ^ (c >> 1)) if (c & 1) else (c >> 1)
        table.append(c)
    _CRC32C_TABLE = table


def crc32c(data: bytes) -> int:
    if _CRC32C_TABLE is None:
        _build_crc32c_table()
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def crc32c_hex(data: bytes) -> str:
    return format(crc32c(data), "08x")


CRC_RE = re.compile(r"^[0-9a-f]{8}$")
GENERATION_RE = re.compile(r"^\d+$")
URI_RE = re.compile(r"^gs://[^/]+/.+$")

# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

WS_RE = re.compile(r"\s+", re.UNICODE)


def canonicalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.strip()
    s = WS_RE.sub(" ", s)
    return s


WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)  # unicode letters/numbers, no underscore


def word_set(s: str) -> set:
    return set(w.lower() for w in WORD_RE.findall(s))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Row shape validation
# ---------------------------------------------------------------------------

ROW_KEYS = {"id", "entity", "eventTime", "revision", "text"}
MAX_SAFE_INT = 2**53 - 1


def row_shape_valid(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if set(row.keys()) != ROW_KEYS:
        return False
    if not isinstance(row["id"], str):
        return False
    if not isinstance(row["entity"], str):
        return False
    if not isinstance(row["eventTime"], str):
        return False
    if not isinstance(row["text"], str):
        return False
    rev = row["revision"]
    if isinstance(rev, bool) or not isinstance(rev, int):
        return False
    if rev < 0 or rev > MAX_SAFE_INT:
        return False
    return True


def compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def utf8_key(s: str) -> bytes:
    return s.encode("utf-8")


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------


@app.post("/build-corpus")
async def build_corpus(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    policy = body.get("policy")
    objects = body.get("objects")

    if not isinstance(policy, dict) or not isinstance(objects, list):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    # --- Policy validity -----------------------------------------------
    min_time_raw = policy.get("minTime")
    max_time_raw = policy.get("maxTime")
    threshold = policy.get("contaminationThreshold")

    min_dt = parse_timestamp(min_time_raw) if isinstance(min_time_raw, str) else None
    max_dt = parse_timestamp(max_time_raw) if isinstance(max_time_raw, str) else None

    threshold_valid = (
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(threshold)
        and 0 <= threshold <= 1
    )

    policy_valid = (min_dt is not None) and (max_dt is not None) and threshold_valid

    # --- Per-object validation ------------------------------------------
    rejected_objects = []
    lineage = []
    retained_rows = []  # each: dict(id, entity, eventTime(raw), revision, text)

    for obj in objects:
        if not isinstance(obj, dict):
            rejected_objects.append(
                {"uri": None, "reasonCodes": sorted(["SCHEMA_INVALID"])}
            )
            continue

        uri = obj.get("uri")
        generation = obj.get("generation")
        fetched_generation = obj.get("fetchedGeneration")
        crc = obj.get("crc32c")
        schema_id = obj.get("schemaId")
        content = obj.get("content")

        codes = set()

        uri_out = uri if isinstance(uri, str) else None
        if not isinstance(uri, str) or not URI_RE.match(uri):
            codes.add("URI_INVALID")

        gen_valid = isinstance(generation, str) and GENERATION_RE.match(generation)
        fetched_valid = isinstance(fetched_generation, str) and GENERATION_RE.match(
            fetched_generation
        )
        if not gen_valid or not fetched_valid:
            codes.add("GENERATION_INVALID")
        elif generation != fetched_generation:
            codes.add("GENERATION_MISMATCH")

        crc_syntax_ok = isinstance(crc, str) and CRC_RE.match(crc)
        if not crc_syntax_ok:
            codes.add("CRC32C_INVALID")
        elif isinstance(content, str):
            actual = crc32c_hex(content.encode("utf-8"))
            if actual != crc:
                codes.add("CRC32C_MISMATCH")

        content_is_str = isinstance(content, str)
        schema_ok = content_is_str and schema_id == "training-v1"

        parsed_rows = []
        jsonl_invalid = False
        schema_invalid_extra = False

        if content_is_str:
            lines = content.split("\n")
            non_blank = [ln for ln in lines if ln.strip() != ""]
            if len(non_blank) == 0:
                schema_invalid_extra = True
            for ln in non_blank:
                try:
                    row = json.loads(ln)
                except Exception:
                    jsonl_invalid = True
                    continue
                if not row_shape_valid(row):
                    schema_invalid_extra = True
                    continue
                parsed_rows.append(row)
        else:
            schema_invalid_extra = True  # non-string content

        if jsonl_invalid:
            codes.add("JSONL_INVALID")
        if not schema_ok or schema_invalid_extra:
            codes.add("SCHEMA_INVALID")

        if codes:
            rejected_objects.append(
                {"uri": uri_out, "reasonCodes": sorted(codes)}
            )
            continue

        # Object accepted.
        lineage.append(
            {
                "uri": uri,
                "generation": generation,
                "crc32c": crc,
                "schemaId": schema_id,
            }
        )
        for row in parsed_rows:
            retained_rows.append(dict(row))

    # --- Canonicalization + eventTime parsing ----------------------------
    working = []
    rejected_rows = []  # list of (id, code)

    for row in retained_rows:
        entity_c = canonicalize_text(row["entity"])
        text_c = canonicalize_text(row["text"])
        dt = parse_timestamp(row["eventTime"])
        if dt is None:
            # ASSUMPTION: an unparseable eventTime cannot be judged to be
            # inside the allowed window, so it is rejected as OUT_OF_WINDOW.
            rejected_rows.append((row["id"], "OUT_OF_WINDOW"))
            continue
        working.append(
            {
                "id": row["id"],
                "entity": entity_c,
                "eventTime": format_utc(dt),
                "eventTime_dt": dt,
                "revision": row["revision"],
                "text": text_c,
            }
        )

    # --- Deduplication -----------------------------------------------
    groups: dict[tuple, list] = {}
    for r in working:
        key = (r["entity"], r["eventTime"], r["text"])
        groups.setdefault(key, []).append(r)

    survivors = []
    for key, rows in groups.items():
        if len(rows) == 1:
            survivors.append(rows[0])
            continue
        best_rev = max(r["revision"] for r in rows)
        candidates = [r for r in rows if r["revision"] == best_rev]
        candidates.sort(key=lambda r: utf8_key(r["id"]))
        winner = candidates[0]
        survivors.append(winner)
        for r in rows:
            if r is not winner:
                rejected_rows.append((r["id"], "DUPLICATE"))

    # --- Policy / window -----------------------------------------------
    windowed = []
    if not policy_valid:
        for r in survivors:
            rejected_rows.append((r["id"], "POLICY_INVALID"))
    else:
        for r in survivors:
            if min_dt <= r["eventTime_dt"] <= max_dt:
                windowed.append(r)
            else:
                rejected_rows.append((r["id"], "OUT_OF_WINDOW"))

    # --- Split assignment -----------------------------------------------
    splits: dict[str, list] = {"train": [], "validation": [], "test": []}
    for r in windowed:
        digest = hashlib.sha256(r["entity"].encode("utf-8")).digest()
        bucket = digest[0] % 10
        if bucket <= 5:
            split = "train"
        elif bucket <= 7:
            split = "validation"
        else:
            split = "test"
        r["_split"] = split
        splits[split].append(r)

    # --- Contamination check -----------------------------------------------
    train_word_sets = [word_set(r["text"]) for r in splits["train"]]

    for split_name in ("validation", "test"):
        keep = []
        for r in splits[split_name]:
            ws = word_set(r["text"])
            contaminated = any(
                jaccard(ws, tws) >= threshold for tws in train_word_sets
            )
            if contaminated:
                rejected_rows.append((r["id"], "TRAIN_CONTAMINATION"))
            else:
                keep.append(r)
        splits[split_name] = keep

    # --- Serialize + hash each split -----------------------------------------------
    def row_public(r):
        return {
            "id": r["id"],
            "entity": r["entity"],
            "eventTime": r["eventTime"],
            "revision": r["revision"],
            "text": r["text"],
        }

    out_splits = {}
    digests = {}
    for split_name in ("train", "validation", "test"):
        rows = splits[split_name]
        rows_sorted = sorted(
            rows, key=lambda r: (utf8_key(r["id"]), compact_json(row_public(r)))
        )
        public_rows = [row_public(r) for r in rows_sorted]
        out_splits[split_name] = public_rows

        buf = bytearray()
        for pr in public_rows:
            line = compact_json(pr) + "\n"
            buf.extend(line.encode("utf-8"))
        digests[split_name] = hashlib.sha256(bytes(buf)).hexdigest()

    # --- Assemble rejected rows / objects / lineage -----------------------------------------------
    row_reason_map: dict[str, set] = {}
    for rid, code in rejected_rows:
        row_reason_map.setdefault(rid, set()).add(code)

    rejected_rows_out = [
        {"id": rid, "reasonCodes": sorted(codes)}
        for rid, codes in row_reason_map.items()
    ]
    rejected_rows_out.sort(key=lambda r: (utf8_key(r["id"]), compact_json(r)))

    rejected_objects.sort(
        key=lambda o: (utf8_key(o["uri"] or ""), compact_json(o))
    )

    lineage.sort(key=lambda o: (utf8_key(o["uri"]), compact_json(o)))

    response = {
        "splits": out_splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows_out,
        "digests": digests,
        "lineage": lineage,
    }
    return JSONResponse(response)


@app.get("/")
async def root():
    return {"status": "ok", "endpoint": "POST /build-corpus"}