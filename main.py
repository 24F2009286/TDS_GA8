from fastapi import FastAPI, Request, Response
import json
import re
import unicodedata
import hashlib
import math
from datetime import datetime, timezone, timedelta

app = FastAPI()

# ==========================================
# SHARED HELPERS
# ==========================================
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
            if hrs > 14 or (hrs == 14 and mins != 0) or mins > 59: return None
            tz = timezone(timedelta(hours=sign*hrs, minutes=sign*mins))
        
        dt = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz)
        if frac: dt = dt.replace(microsecond=int(frac.ljust(6, '0')))
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None

def _reject_constant(_s):
    raise ValueError("non-JSON constant")


# ==========================================
# Q1: BUILD CORPUS LOGIC
# ==========================================
CRC32C_TABLE = [0] * 256
for i in range(256):
    c = i
    for _ in range(8): c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
    CRC32C_TABLE[i] = c

def compute_crc32c(data: bytes) -> str:
    crc = 0xFFFFFFFF
    for b in data: crc = CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return f"{(crc ^ 0xFFFFFFFF):08x}"

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
        "id": row["id"], "entity": row["_norm_entity"],
        "eventTime": row["_norm_time"], "revision": row["revision"],
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
    rejected_objects_list, rejected_rows_list, valid_rows, lineage = [], [], [], []
    global_order = 0

    for obj in objects:
        if type(obj) is not dict: obj = {}
        uri = obj.get("uri")
        obj_reasons = set()
        uri_val = uri if type(uri) is str else None

        if type(uri) is not str or not re.fullmatch(r'^gs://[^/]+/.+$', uri, flags=re.DOTALL):
            obj_reasons.add("URI_INVALID")
            
        gen, f_gen = obj.get("generation"), obj.get("fetchedGeneration")
        gen_valid = type(gen) is str and re.fullmatch(r'^[0-9]+$', gen) is not None
        f_gen_valid = type(f_gen) is str and re.fullmatch(r'^[0-9]+$', f_gen) is not None
        
        if not gen_valid or not f_gen_valid: obj_reasons.add("GENERATION_INVALID")
        if gen != f_gen: obj_reasons.add("GENERATION_MISMATCH")

        crc = obj.get("crc32c")
        crc_valid = type(crc) is str and re.fullmatch(r'^[0-9a-f]{8}$', crc)
        if not crc_valid: obj_reasons.add("CRC32C_INVALID")

        content, schema = obj.get("content"), obj.get("schemaId")
        has_schema_error, has_jsonl_error, parsed_rows = False, False, []

        if schema != "training-v1": has_schema_error = True
            
        if type(content) is not str:
            has_schema_error = True
        else:
            if crc_valid and compute_crc32c(content.encode('utf-8')) != crc:
                obj_reasons.add("CRC32C_MISMATCH")

            lines = [l[:-1] if l.endswith('\r') else l for l in content.split('\n')]
            non_blank_lines = [l for l in lines if l.strip()]
            if not non_blank_lines: has_schema_error = True
            
            for line in non_blank_lines:
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                    if type(row) is not dict or set(row.keys()) != {"id", "entity", "eventTime", "revision", "text"}:
                        has_schema_error = True; continue
                    if not all(type(row[k]) is str for k in ["id", "entity", "eventTime", "text"]):
                        has_schema_error = True; continue
                    rev = row["revision"]
                    if type(rev) is not int or type(rev) is bool or rev < 0 or rev > 9007199254740991:
                        has_schema_error = True; continue
                    
                    dt = parse_and_validate_time(row["eventTime"])
                    if dt is None: has_schema_error = True; continue
                        
                    row["_parsed_dt"], row["_order"] = dt, global_order
                    global_order += 1
                    parsed_rows.append(row)
                except Exception:
                    has_jsonl_error = True

        if has_jsonl_error: obj_reasons.add("JSONL_INVALID")
        if has_schema_error: obj_reasons.add("SCHEMA_INVALID")

        if obj_reasons:
            rejected_objects_list.append({"uri": uri_val, "reasonCodes": sorted(list(obj_reasons), key=lambda x: x.encode('utf-8'))})
        else:
            lineage.append({"uri": uri, "generation": gen, "crc32c": crc, "schemaId": schema})
            valid_rows.extend(parsed_rows)

    groups = {}
    for row in valid_rows:
        norm_time = row["_parsed_dt"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        norm_entity, norm_text = normalize_text(row["entity"]), normalize_text(row["text"])
        
        row.update({"_norm_time": norm_time, "_norm_entity": norm_entity, "_norm_text": norm_text, "_raw_id_bytes": row["id"].encode('utf-8'), "_words": extract_words(norm_text)})
        tup = (norm_entity, norm_time, norm_text)
        if tup not in groups: groups[tup] = []
        groups[tup].append(row)

    retained_rows = []
    for key, members in groups.items():
        members_sorted = sorted(members, key=lambda r: (-r["revision"], r["_raw_id_bytes"], r["_order"]))
        retained_rows.append(members_sorted[0])
        for loser in members_sorted[1:]: rejected_rows_list.append({"id": loser["id"], "reasonCodes": ["DUPLICATE"]})

    retained_rows.sort(key=lambda r: r["_order"])
    window_survivors = []
    
    if not policy_valid:
        for row in retained_rows: rejected_rows_list.append({"id": row["id"], "reasonCodes": ["POLICY_INVALID"]})
    else:
        for row in retained_rows:
            if policy_min <= row["_parsed_dt"] <= policy_max: window_survivors.append(row)
            else: rejected_rows_list.append({"id": row["id"], "reasonCodes": ["OUT_OF_WINDOW"]})

    train_pool, val_test_pool = [], []
    for row in window_survivors:
        bucket = hashlib.sha256(row["_norm_entity"].encode('utf-8')).digest()[0] % 10
        row["_split"] = "train" if bucket <= 5 else "validation" if bucket <= 7 else "test"
        (train_pool if bucket <= 5 else val_test_pool).append(row)

    splits = {"train": train_pool, "validation": [], "test": []}
    for row in val_test_pool:
        if any(jaccard(row["_words"], tr["_words"]) >= thresh for tr in train_pool):
            rejected_rows_list.append({"id": row["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
        else: splits[row["_split"]].append(row)

    formatted_splits, digests = {"train": [], "validation": [], "test": []}, {"train": "", "validation": "", "test": ""}
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

    return Response(content=json.dumps({
        "splits": formatted_splits, "rejectedObjects": rejected_objects_list,
        "rejectedRows": rejected_rows_list, "digests": digests, "lineage": lineage
    }, ensure_ascii=False, separators=(',', ':')), status_code=200, media_type="application/json")


# ==========================================
# Q2: BIGQUERY ML LOGIC
# ==========================================
_STATE = {}

def sha256_compact(data: dict) -> str:
    compact_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(compact_str.encode('utf-8')).hexdigest()

def handle_select(payload: dict):
    run_id = payload.get("runId")
    if type(run_id) is not str or not run_id or len(run_id) > 128: return None, ["INVALID_INPUT"]
    
    if run_id in _STATE:
        return (_STATE[run_id]["response"], None) if _STATE[run_id]["request"] == payload else ("CONFLICT", None)

    limit = payload.get("numTrialsLimit")
    if type(limit) is not int or limit <= 0: return None, ["INVALID_INPUT"]

    rows, trials = payload.get("rows"), payload.get("trials")
    if type(rows) is not list or not rows or type(trials) is not list: return None, ["INVALID_INPUT"]

    groups = {}
    for r in rows:
        if type(r) is not dict: return None, ["INVALID_INPUT"]
        r_id, ent, evt, pt, ver, split, feats = r.get("id"), r.get("entity"), r.get("eventTime"), r.get("predictionTime"), r.get("version"), r.get("split"), r.get("features")
        
        if not all(type(x) is str for x in [r_id, ent, evt, pt, split]) or type(ver) is not int or ver < 0 or split not in ("TRAIN", "EVAL") or type(feats) is not dict: return None, ["INVALID_INPUT"]
        
        evt_dt, pt_dt = parse_and_validate_time(evt), parse_and_validate_time(pt)
        if not evt_dt or not pt_dt: return None, ["INVALID_INPUT"]
        
        r.update({"_evt_dt": evt_dt, "_pt_dt": pt_dt, "_id_bytes": r_id.encode('utf-8')})
        key = (ent, evt_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
        if key not in groups: groups[key] = []
        groups[key].append(r)

    retained = [sorted(members, key=lambda x: (-x["version"], x["_id_bytes"]))[0] for members in groups.values()]
    if not retained: return None, ["INVALID_INPUT"]

    forbidden = set(payload.get("forbiddenFeatures", []))
    feature_counts, feature_valid = {}, {}
    
    for r in retained:
        for fname, fval in r["features"].items():
            if fname in forbidden: continue
            if type(fval) is not dict: return None, ["INVALID_INPUT"]
            avail_dt = parse_and_validate_time(fval.get("availableAt", ""))
            if not avail_dt: return None, ["INVALID_INPUT"]
            
            feature_counts[fname] = feature_counts.get(fname, 0) + 1
            if fname not in feature_valid: feature_valid[fname] = True
            if avail_dt > r["_pt_dt"]: feature_valid[fname] = False

    valid_features = sorted([fname for fname, count in feature_counts.items() if count == len(retained) and feature_valid[fname]], key=lambda x: x.encode('utf-8'))

    if len(trials) > limit: return None, ["TRIAL_LIMIT_EXCEEDED"]
        
    best_trial = None
    for t in trials:
        if type(t) is not dict: return None, ["INVALID_INPUT"]
        t_id, status, metric = t.get("trialId"), t.get("status"), t.get("evalMetric")
        
        if type(t_id) is not int or t_id < 0 or status not in ("SUCCEEDED", "FAILED") or type(metric) not in (float, int) or not math.isfinite(metric): return None, ["INVALID_INPUT"]
        
        if status == "SUCCEEDED":
            if best_trial is None or metric > best_trial["metric"] or (metric == best_trial["metric"] and t_id < best_trial["id"]):
                best_trial = {"id": t_id, "metric": metric}

    if not best_trial: return None, ["NO_SUCCESSFUL_TRIAL"]

    train_ids = sorted([r["id"] for r in retained if r["split"] == "TRAIN"], key=lambda x: x.encode('utf-8'))
    eval_ids = sorted([r["id"] for r in retained if r["split"] == "EVAL"], key=lambda x: x.encode('utf-8'))
    
    response = {
        "runId": run_id, "selectedTrialId": best_trial["id"], "trainRowIds": train_ids,
        "evalRowIds": eval_ids, "featureNames": valid_features,
        "datasetDigest": sha256_compact({"trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": valid_features}),
        "reasonCodes": []
    }
    
    _STATE[run_id] = {"request": payload, "response": response}
    return response, None

def handle_evaluate(payload: dict):
    run_id, trial_id, digest = payload.get("runId"), payload.get("selectedTrialId"), payload.get("datasetDigest")
    if type(run_id) is not str or type(trial_id) is not int or type(digest) is not str: return None, ["INVALID_INPUT"], False
        
    codes, test_metric, critical_pass = set(), None, True
    
    if run_id not in _STATE or _STATE[run_id]["response"]["selectedTrialId"] != trial_id or _STATE[run_id]["response"]["datasetDigest"] != digest:
        codes.add("INVALID_LINEAGE"); critical_pass = False

    floor_agg, req_slices, bp, mb, rows = payload.get("metricFloor"), payload.get("requiredSlices"), payload.get("bytesProcessed"), payload.get("maxBytes"), payload.get("rows")
    
    if type(floor_agg) not in (float, int) or not (0 <= floor_agg <= 1) or type(req_slices) is not dict or type(bp) is not int or bp < 0 or type(mb) is not int or mb < 0 or type(rows) is not list:
        codes.add("INVALID_INPUT")

    if "INVALID_INPUT" in codes: return None, list(codes), False
    if bp > mb: codes.add("BYTE_LIMIT")

    valid_rows, slice_stats, total_correct, total_rows = True, {}, 0, 0
    if not rows: valid_rows, critical_pass = False, False
        
    for r in rows:
        if type(r) is not dict: codes.add("INVALID_TEST_ROW"); valid_rows = False; continue
        lbl, prd, slc = r.get("label"), r.get("prediction"), r.get("slice")
        if lbl not in (0, 1) or prd not in (0, 1) or type(slc) is not str or not slc:
            codes.add("INVALID_TEST_ROW"); valid_rows = False; continue
            
        total_rows += 1
        is_correct = (lbl == prd)
        if is_correct: total_correct += 1
        
        if slc not in slice_stats: slice_stats[slc] = {"c": 0, "t": 0}
        slice_stats[slc]["t"] += 1
        if is_correct: slice_stats[slc]["c"] += 1

    if not valid_rows:
        critical_pass = False
    else:
        test_metric = round(total_correct / total_rows, 12)
        if test_metric < floor_agg: codes.add("AGGREGATE_FLOOR")
            
        for req_slice, s_floor in req_slices.items():
            if type(s_floor) not in (float, int) or not (0 <= s_floor <= 1):
                codes.add("INVALID_INPUT"); critical_pass = False; continue
            if req_slice not in slice_stats:
                codes.add(f"MISSING_SLICE:{req_slice}"); critical_pass = False
            elif round(slice_stats[req_slice]["c"] / slice_stats[req_slice]["t"], 12) < s_floor:
                codes.add(f"SLICE_FLOOR:{req_slice}"); critical_pass = False

    return {
        "runId": run_id, "selectedTrialId": trial_id, "datasetDigest": digest, "testMetric": test_metric,
        "criticalSlicePass": critical_pass, "decision": "admit" if (not codes and test_metric is not None and critical_pass) else "reject",
        "bytesProcessed": bp, "reasonCodes": sorted(list(codes), key=lambda x: x.encode('utf-8')) if codes else []
    }, None, None

@app.post("/bqml")
async def bqml_endpoint(request: Request):
    try: payload = json.loads(await request.body())
    except Exception: return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")
        
    if type(payload) is not dict: return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    phase = payload.get("phase")
    if phase == "select":
        res, errs = handle_select(payload)
        if errs:
            if errs[0] == "CONFLICT": return Response(content=json.dumps({"error": "RUN_ID_CONFLICT"}), status_code=409, media_type="application/json")
            return Response(content=json.dumps({
                "runId": payload.get("runId") if type(payload.get("runId")) is str else "", "selectedTrialId": None,
                "trainRowIds": [], "evalRowIds": [], "featureNames": [], "datasetDigest": None, "reasonCodes": sorted(errs, key=lambda x: x.encode('utf-8'))
            }), status_code=200, media_type="application/json")
        return Response(content=json.dumps(res), status_code=200, media_type="application/json")

    elif phase == "evaluate":
        res, errs, _ = handle_evaluate(payload)
        if errs and res is None:
            return Response(content=json.dumps({
                "runId": payload.get("runId") if type(payload.get("runId")) is str else "",
                "selectedTrialId": payload.get("selectedTrialId") if type(payload.get("selectedTrialId")) is int else 0,
                "datasetDigest": payload.get("datasetDigest") if type(payload.get("datasetDigest")) is str else "",
                "testMetric": None, "criticalSlicePass": False, "decision": "reject",
                "bytesProcessed": payload.get("bytesProcessed") if type(payload.get("bytesProcessed")) is int else 0,
                "reasonCodes": sorted(errs, key=lambda x: x.encode('utf-8'))
            }), status_code=200, media_type="application/json")
        return Response(content=json.dumps(res), status_code=200, media_type="application/json")
    else:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")