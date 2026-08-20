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
    match = re.fullmatch(r'^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})(?:\.([0-9]{1,3}))?(Z|[+-][0-9]{2}:[0-9]{2})$', ts)
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
    return set(re.findall(r'[^\W_]+', text, flags=re.UNICODE))

def jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def generate_compact_json(row: dict) -> bytes:
    ordered = {
        "id": row["id"],
        "entity": row["_norm_entity"],
        "eventTime": row["_norm_time"],
        "revision": row["revision"],
        "text": row["_norm_text"]
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

@app.post("/build-corpus")
@app.post("/build-corpus/")
async def build_corpus(request: Request):
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes)
    except Exception:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")
        
    # Strictly trap null policies and non-arrays to satisfy 400 Bad Request limits
    if type(payload) is not dict or payload.get("policy") is None or type(payload.get("objects")) is not list:
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
    rejected_objects_list = []
    valid_rows = []
    lineage = []

    # 1. Object Identity and Integrity Pipeline
    for obj in objects:
        if type(obj) is not dict:
            obj = {}
            
        uri = obj.get("uri")
        obj_reasons = set()
        uri_val = uri if type(uri) is str else None

        # re.DOTALL ensures malicious internal newlines in the object name still match the valid URI schema
        if type(uri) is not str or not re.fullmatch(r'^gs://[^/]+/.+$', uri, flags=re.DOTALL):
            obj_reasons.add("URI_INVALID")
            
        gen = obj.get("generation")
        f_gen = obj.get("fetchedGeneration")
        
        gen_valid = type(gen) is str and re.fullmatch(r'^[0-9]+$', gen) is not None
        f_gen_valid = type(f_gen) is str and re.fullmatch(r'^[0-9]+$', f_gen) is not None
        
        if not gen_valid or not f_gen_valid:
            obj_reasons.add("GENERATION_INVALID")
            
        if "generation" in obj and "fetchedGeneration" in obj:
            if gen != f_gen:
                obj_reasons.add("GENERATION_MISMATCH")

        crc = obj.get("crc32c")
        crc_valid = type(crc) is str and re.fullmatch(r'^[0-9a-f]{8}$', crc)
        if not crc_valid:
            obj_reasons.add("CRC32C_INVALID")

        content = obj.get("content")
        schema = obj.get("schemaId")
        
        has_schema_error = False
        has_jsonl_error = False
        parsed_rows = []

        if schema != "training-v1":
            has_schema_error = True
            
        if type(content) is not str:
            has_schema_error = True
        else:
            if crc_valid:
                calculated_crc = compute_crc32c(content.encode('utf-8'))
                if calculated_crc != crc:
                    obj_reasons.add("CRC32C_MISMATCH")

            lines = content.split('\n')
            non_blank_lines = [l for l in lines if l.strip()]
            if not non_blank_lines:
                has_schema_error = True
            
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
                    row["reasonCodes"] = set()
                    parsed_rows.append(row)
                except Exception:
                    has_jsonl_error = True

        if has_jsonl_error:
            obj_reasons.add("JSONL_INVALID")
        if has_schema_error:
            obj_reasons.add("SCHEMA_INVALID")

        if obj_reasons:
            rejected_objects_list.append({
                "uri": uri_val,
                "reasonCodes": sorted(list(obj_reasons), key=lambda x: x.encode('utf-8'))
            })
        else:
            lineage.append({
                "uri": uri,
                "generation": gen,
                "crc32c": crc,
                "schemaId": schema
            })
            for r in parsed_rows:
                valid_rows.append(r)

    # 2. Canonicalization and Deduplication Pipeline
    dedup_map = {}
    for row in valid_rows:
        dt = row["_parsed_dt"]
        norm_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        norm_entity = normalize_text(row["entity"])
        norm_text = normalize_text(row["text"])
        
        row["_norm_time"] = norm_time
        row["_norm_entity"] = norm_entity
        row["_norm_text"] = norm_text
        row["_raw_id_bytes"] = row["id"].encode('utf-8')
        row["_words"] = extract_words(norm_text)
        
        tup = (norm_entity, norm_time, norm_text)

        if tup not in dedup_map:
            dedup_map[tup] = row
        else:
            existing = dedup_map[tup]
            if row["revision"] > existing["revision"] or \
               (row["revision"] == existing["revision"] and row["_raw_id_bytes"] < existing["_raw_id_bytes"]):
                existing["reasonCodes"].add("DUPLICATE")
                dedup_map[tup] = row
            else:
                row["reasonCodes"].add("DUPLICATE")

    # 3. Policy Window Check
    retained_rows = list(dedup_map.values())
    window_survivors = []
    
    for row in retained_rows:
        if not policy_valid:
            row["reasonCodes"].add("POLICY_INVALID")
        else:
            if policy_min <= row["_parsed_dt"] <= policy_max:
                window_survivors.append(row)
            else:
                row["reasonCodes"].add("OUT_OF_WINDOW")

    # 4. Routing and Contamination Check
    train_pool = []
    val_test_pool = []

    for row in window_survivors:
        entity_bytes = row["_norm_entity"].encode('utf-8')
        first_byte = hashlib.sha256(entity_bytes).digest()[0]
        bucket = first_byte % 10
        
        if 0 <= bucket <= 5:
            row["_split"] = "train"
            train_pool.append(row)
        elif 6 <= bucket <= 7:
            row["_split"] = "validation"
            val_test_pool.append(row)
        else:
            row["_split"] = "test"
            val_test_pool.append(row)

    splits = {"train": train_pool, "validation": [], "test": []}
        
    for row in val_test_pool:
        contaminated = False
        for tr in train_pool:
            if jaccard(row["_words"], tr["_words"]) >= thresh:
                contaminated = True
                break
        
        if contaminated:
            row["reasonCodes"].add("TRAIN_CONTAMINATION")
        else:
            splits[row["_split"]].append(row)

    # 5. Build Final Data Structures
    rejected_rows_list = []
    for row in valid_rows:
        if row["reasonCodes"]:
            rejected_rows_list.append({
                "id": row["id"],
                "reasonCodes": sorted(list(row["reasonCodes"]), key=lambda x: x.encode('utf-8'))
            })

    formatted_splits = {"train": [], "validation": [], "test": []}
    digests = {"train": "", "validation": "", "test": ""}

    for k in splits:
        splits[k].sort(key=lambda x: (x["_raw_id_bytes"], generate_compact_json(x)))
        raw_bytes = b""
        for r in splits[k]:
            compact = generate_compact_json(r)
            formatted_splits[k].append(json.loads(compact.decode('utf-8')))
            raw_bytes += compact + b"\n"
        digests[k] = hashlib.sha256(raw_bytes).hexdigest()

    rejected_objects_list.sort(key=lambda x: (x["uri"].encode('utf-8') if type(x["uri"]) is str else b"", json.dumps(x, separators=(',', ':')).encode('utf-8')))
    rejected_rows_list.sort(key=lambda x: (x["id"].encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))
    lineage.sort(key=lambda x: (x["uri"].encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))

    response_payload = {
        "splits": formatted_splits,
        "rejectedObjects": rejected_objects_list,
        "rejectedRows": rejected_rows_list,
        "digests": digests,
        "lineage": lineage
    }

    return Response(content=json.dumps(response_payload, ensure_ascii=False, separators=(',', ':')), status_code=200, media_type="application/json")