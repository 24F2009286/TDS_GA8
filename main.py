from fastapi import FastAPI, Request, Response
import json
import re
import unicodedata
import hashlib
from datetime import datetime, timezone, timedelta

app = FastAPI()

# Precompute CRC32C table for pure-Python execution without C-extensions
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
    if type(ts) is not str: return None
    match = re.fullmatch(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$', ts)
    if not match: return None
    
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
    # [^\W_]+ strictly extracts Unicode letters and numbers, excluding underscores and punctuation
    return set(re.findall(r'[^\W_]+', text, flags=re.UNICODE))

def jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def generate_compact_json(row: dict) -> bytes:
    ordered = {
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"]
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

@app.post("/build-corpus")
@app.post("/build-corpus/")
async def build_corpus(request: Request):
    # Raw byte parsing to prevent FastAPI from throwing 422 on bad JSON bodies
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes)
    except Exception:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")
        
    if type(payload) is not dict or "policy" not in payload or type(payload.get("objects")) is not list:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    policy = payload["policy"]
    policy_valid = False
    if type(policy) is dict:
        policy_min = parse_and_validate_time(policy.get("minTime", ""))
        policy_max = parse_and_validate_time(policy.get("maxTime", ""))
        thresh = policy.get("contaminationThreshold")
        if policy_min and policy_max and type(thresh) in (int, float) and not type(thresh) is bool and 0.0 <= thresh <= 1.0:
            policy_valid = True

    objects = payload["objects"]
    rejected_objects = {}
    rejected_rows = {}
    valid_rows = []
    lineage = []

    # 1. Object Identity and Integrity Pipeline
    for obj in objects:
        if type(obj) is not dict: continue
        
        uri = obj.get("uri")
        obj_reasons = set()
        uri_val = uri if type(uri) is str else None

        if type(uri) is not str or not re.match(r'^gs://[^/]+/.+$', uri):
            obj_reasons.add("URI_INVALID")
            
        gen = obj.get("generation")
        f_gen = obj.get("fetchedGeneration")
        
        gen_valid = type(gen) is str and gen.isdecimal()
        f_gen_valid = type(f_gen) is str and f_gen.isdecimal()
        
        if not gen_valid or not f_gen_valid:
            obj_reasons.add("GENERATION_INVALID")
        elif gen != f_gen:
            obj_reasons.add("GENERATION_MISMATCH")

        crc = obj.get("crc32c")
        crc_valid = type(crc) is str and re.fullmatch(r'^[0-9a-f]{8}$', crc)
        if not crc_valid:
            obj_reasons.add("CRC32C_INVALID")

        content = obj.get("content")
        schema = obj.get("schemaId")
        
        if schema != "training-v1":
            obj_reasons.add("SCHEMA_INVALID")
            
        parsed_rows = []
        if type(content) is not str:
            obj_reasons.add("SCHEMA_INVALID")
        else:
            if crc_valid:
                calculated_crc = compute_crc32c(content.encode('utf-8'))
                if calculated_crc != crc:
                    obj_reasons.add("CRC32C_MISMATCH")

            lines = content.split('\n')
            non_blank_lines = [l for l in lines if l.strip()]
            if not non_blank_lines:
                obj_reasons.add("SCHEMA_INVALID")
            
            has_jsonl_error = False
            has_schema_error = False
            
            for line in non_blank_lines:
                try:
                    row = json.loads(line)
                    if type(row) is not dict:
                        has_schema_error = True
                        continue
                    if set(row.keys()) != {"id", "entity", "eventTime", "revision", "text"}:
                        has_schema_error = True
                        continue
                    if not all(type(row[k]) is str for k in ["id", "entity", "eventTime", "text"]):
                        has_schema_error = True
                        continue
                    rev = row["revision"]
                    if type(rev) is not int or type(rev) is bool or rev < 0 or rev > 9007199254740991:
                        has_schema_error = True
                        continue
                    
                    dt = parse_and_validate_time(row["eventTime"])
                    if dt is None:
                        has_schema_error = True
                        continue
                        
                    row["_parsed_dt"] = dt 
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

    # 2. Canonicalization and Deduplication
    dedup_map = {}
    for row in valid_rows:
        dt = row["_parsed_dt"]
        norm_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        norm_entity = normalize_text(row["entity"])
        norm_text = normalize_text(row["text"])
        
        tup = (norm_entity, norm_time, norm_text)
        candidate = {
            "id": row["id"],
            "entity": norm_entity,
            "eventTime": norm_time,
            "revision": row["revision"],
            "text": norm_text,
            "dt": dt,
            "raw_id_bytes": row["id"].encode('utf-8')
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
                if candidate["id"] not in rejected_rows: rejected_rows[candidate["id"]] = set()
                rejected_rows[candidate["id"]].add("DUPLICATE")

    # 3. Policy Window Check
    retained = list(dedup_map.values())
    if not policy_valid:
        for r in retained:
            if r["id"] not in rejected_rows: rejected_rows[r["id"]] = set()
            rejected_rows[r["id"]].add("POLICY_INVALID")
        retained = []
    else:
        window_retained = []
        for r in retained:
            if policy_min <= r["dt"] <= policy_max:
                window_retained.append(r)
            else:
                if r["id"] not in rejected_rows: rejected_rows[r["id"]] = set()
                rejected_rows[r["id"]].add("OUT_OF_WINDOW")
        retained = window_retained

    # 4. Routing and Contamination Check
    train_pool = []
    val_test_pool = []

    for r in retained:
        entity_bytes = r["entity"].encode('utf-8')
        first_byte = hashlib.sha256(entity_bytes).digest()[0]
        bucket = first_byte % 10
        r["words"] = extract_words(r["text"])
        
        if 0 <= bucket <= 5:
            r["split"] = "train"
            train_pool.append(r)
        elif 6 <= bucket <= 7:
            r["split"] = "validation"
            val_test_pool.append(r)
        else:
            r["split"] = "test"
            val_test_pool.append(r)

    splits = {"train": train_pool, "validation": [], "test": []}
        
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

    # 5. Serialization and Hashing
    formatted_splits = {"train": [], "validation": [], "test": []}
    digests = {"train": "", "validation": "", "test": ""}

    for k in splits:
        splits[k].sort(key=lambda x: (x["raw_id_bytes"], generate_compact_json(x)))
        raw_bytes = b""
        for r in splits[k]:
            compact = generate_compact_json(r)
            formatted_splits[k].append(json.loads(compact.decode('utf-8')))
            raw_bytes += compact + b"\n"
        digests[k] = hashlib.sha256(raw_bytes).hexdigest()

    # 6. Sorting exact output structures
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