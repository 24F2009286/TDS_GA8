from fastapi import FastAPI, Request, Response
import json
import re
import unicodedata
import hashlib
from datetime import datetime, timezone, timedelta

app = FastAPI()

# Precompute CRC32C table for pure-Python execution
CRC32C_TABLE = [0] * 256
for i in range(256):
    c = i
    for _ in range(8):
        c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
    CRC32C_TABLE[i] = c

def compute_crc32c(data: bytes) -> str:
    crc = 0xFFFFFFFF
    for b in data:
        crc = CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return f"{(crc ^ 0xFFFFFFFF):08x}"

def parse_and_validate_time(ts: str):
    match = re.fullmatch(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$', ts)
    if not match:
        return None
    
    base, frac, offset_str = match.groups()
    
    try:
        if offset_str == 'Z':
            tz = timezone.utc
        else:
            sign = 1 if offset_str[0] == '+' else -1
            hrs, mins = map(int, offset_str[1:].split(':'))
            if hrs > 14 or (hrs == 14 and mins != 0) or mins > 59:
                return None
            tz = timezone(timedelta(hours=sign*hrs, minutes=sign*mins))
        
        dt = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz)
        if frac:
            dt = dt.replace(microsecond=int(frac.ljust(6, '0')))
        
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None

def normalize_text(text: str) -> str:
    nfkc = unicodedata.normalize('NFKC', text).lower()
    return re.sub(r'\s+', ' ', nfkc).strip()

def extract_words(text: str) -> set:
    return set(re.findall(r'[^\W_]+', text, flags=re.UNICODE))

def jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def generate_compact_json(row: dict) -> bytes:
    # Exact key order: id, entity, eventTime, revision, text
    ordered = {
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"]
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

@app.post("/build-corpus")
async def build_corpus(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")
        
    if not isinstance(payload, dict) or "policy" not in payload or not isinstance(payload.get("objects"), list):
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    policy = payload["policy"]
    if not isinstance(policy, dict):
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    policy_min = parse_and_validate_time(policy.get("minTime", ""))
    policy_max = parse_and_validate_time(policy.get("maxTime", ""))
    thresh = policy.get("contaminationThreshold")
    
    policy_valid = (
        policy_min is not None and 
        policy_max is not None and 
        isinstance(thresh, (int, float)) and 
        not isinstance(thresh, bool) and 
        0.0 <= thresh <= 1.0
    )

    objects = payload["objects"]
    
    rejected_objects = {}
    rejected_rows = {}
    valid_rows = []
    lineage = []

    for obj in objects:
        if not isinstance(obj, dict): continue
        
        uri = obj.get("uri")
        obj_reasons = set()
        
        uri_val = uri if isinstance(uri, str) else None

        if not isinstance(uri, str) or not re.match(r'^gs://[^/]+/.+$', uri):
            obj_reasons.add("URI_INVALID")
            
        gen = obj.get("generation")
        f_gen = obj.get("fetchedGeneration")
        
        gen_valid = isinstance(gen, str) and gen.isdecimal()
        f_gen_valid = isinstance(f_gen, str) and f_gen.isdecimal()
        
        if not gen_valid or not f_gen_valid:
            obj_reasons.add("GENERATION_INVALID")
        elif gen != f_gen:
            obj_reasons.add("GENERATION_MISMATCH")

        crc = obj.get("crc32c")
        crc_valid = isinstance(crc, str) and re.fullmatch(r'^[0-9a-f]{8}$', crc)
        if not crc_valid:
            obj_reasons.add("CRC32C_INVALID")

        content = obj.get("content")
        schema = obj.get("schemaId")
        
        if schema != "training-v1":
            obj_reasons.add("SCHEMA_INVALID")
            
        if not isinstance(content, str):
            obj_reasons.add("SCHEMA_INVALID")
        else:
            if crc_valid:
                calculated_crc = compute_crc32c(content.encode('utf-8'))
                if calculated_crc != crc:
                    obj_reasons.add("CRC32C_MISMATCH")

            lines = content.split('\n')
            parsed_rows = []
            has_jsonl_error = False
            has_schema_error = False
            
            non_blank_lines = [l for l in lines if l.strip()]
            if not non_blank_lines:
                obj_reasons.add("SCHEMA_INVALID")
            
            for line in non_blank_lines:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        has_schema_error = True
                        continue
                    
                    if set(row.keys()) != {"id", "entity", "eventTime", "revision", "text"}:
                        has_schema_error = True
                        continue
                        
                    if not all(isinstance(row[k], str) for k in ["id", "entity", "eventTime", "text"]):
                        has_schema_error = True
                        continue
                        
                    rev = row["revision"]
                    if not isinstance(rev, int) or isinstance(rev, bool) or rev < 0:
                        has_schema_error = True
                        continue
                        
                    parsed_rows.append(row)
                except Exception:
                    has_jsonl_error = True

            if has_jsonl_error:
                obj_reasons.add("JSONL_INVALID")
            if has_schema_error:
                obj_reasons.add("SCHEMA_INVALID")

        if obj_reasons:
            rejected_objects[uri_val] = obj_reasons
        else:
            lineage.append({
                "uri": uri,
                "generation": gen,
                "crc32c": crc,
                "schemaId": schema
            })
            for r in parsed_rows:
                valid_rows.append(r)

    # Row Normalization & Deduplication
    dedup_map = {}
    
    for row in valid_rows:
        row_id = row["id"]
        
        dt = parse_and_validate_time(row["eventTime"])
        if not dt:
            if row_id not in rejected_rows: rejected_rows[row_id] = set()
            rejected_rows[row_id].add("SCHEMA_INVALID")
            continue

        norm_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        norm_entity = normalize_text(row["entity"])
        norm_text = normalize_text(row["text"])
        
        if not policy_valid:
            if row_id not in rejected_rows: rejected_rows[row_id] = set()
            rejected_rows[row_id].add("POLICY_INVALID")
            continue
            
        if not (policy_min <= dt <= policy_max):
            if row_id not in rejected_rows: rejected_rows[row_id] = set()
            rejected_rows[row_id].add("OUT_OF_WINDOW")
            continue

        tup = (norm_entity, norm_time, norm_text)
        candidate = {
            "id": row_id,
            "entity": norm_entity,
            "eventTime": norm_time,
            "revision": row["revision"],
            "text": norm_text,
            "raw_id_bytes": row_id.encode('utf-8')
        }

        if tup not in dedup_map:
            dedup_map[tup] = candidate
        else:
            existing = dedup_map[tup]
            if candidate["revision"] > existing["revision"] or \
               (candidate["revision"] == existing["revision"] and candidate["raw_id_bytes"] < existing["raw_id_bytes"]):
                if existing["id"] not in rejected_rows: rejected_rows[existing["id"]] = set()
                rejected_rows[existing["id"]].add("DUPLICATE")
                dedup_map[tup] = candidate
            else:
                if row_id not in rejected_rows: rejected_rows[row_id] = set()
                rejected_rows[row_id].add("DUPLICATE")

    # Routing and Contamination
    train_pool = []
    val_test_pool = []

    for row in dedup_map.values():
        entity_bytes = row["entity"].encode('utf-8')
        first_byte = hashlib.sha256(entity_bytes).digest()[0]
        bucket = first_byte % 10
        
        row["words"] = extract_words(row["text"])
        
        if 0 <= bucket <= 5:
            train_pool.append(row)
        elif 6 <= bucket <= 7:
            row["split"] = "validation"
            val_test_pool.append(row)
        else:
            row["split"] = "test"
            val_test_pool.append(row)

    splits = {"train": [], "validation": [], "test": []}
    for r in train_pool:
        splits["train"].append(r)
        
    for r in val_test_pool:
        contaminated = False
        for tr in train_pool:
            if jaccard(r["words"], tr["words"]) >= thresh:
                contaminated = True
                break
        
        if contaminated:
            if r["id"] not in rejected_rows: rejected_rows[r["id"]] = set()
            rejected_rows[r["id"]].add("TRAIN_CONTAMINATION")
        else:
            splits[r["split"]].append(r)

    # Formatting and Sorting
    for k in splits:
        splits[k].sort(key=lambda x: (x["raw_id_bytes"], generate_compact_json(x)))
        for r in splits[k]:
            del r["words"]
            del r["raw_id_bytes"]
            if "split" in r: del r["split"]

    formatted_splits = {"train": [], "validation": [], "test": []}
    digests = {"train": "", "validation": "", "test": ""}

    for k in splits:
        raw_bytes = b""
        for r in splits[k]:
            compact = generate_compact_json(r)
            formatted_splits[k].append(json.loads(compact.decode('utf-8')))
            raw_bytes += compact + b"\n"
        digests[k] = hashlib.sha256(raw_bytes).hexdigest()

    rej_objs_list = [{"uri": k, "reasonCodes": sorted(list(v))} for k, v in rejected_objects.items()]
    rej_objs_list.sort(key=lambda x: (x["uri"].encode('utf-8') if x["uri"] else b"", json.dumps(x, separators=(',', ':')).encode('utf-8')))
    
    rej_rows_list = [{"id": k, "reasonCodes": sorted(list(v))} for k, v in rejected_rows.items()]
    rej_rows_list.sort(key=lambda x: (x["id"].encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))

    lineage.sort(key=lambda x: (x["uri"].encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))

    response_payload = {
        "splits": formatted_splits,
        "rejectedObjects": rej_objs_list,
        "rejectedRows": rej_rows_list,
        "digests": digests,
        "lineage": lineage
    }

    return Response(content=json.dumps(response_payload, ensure_ascii=False, separators=(',', ':')), status_code=200, media_type="application/json")