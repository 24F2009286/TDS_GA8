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

def is_safe_int(v, non_negative=False):
    if type(v) is not int: return False
    if non_negative and v < 0: return False
    return -9007199254740991 <= v <= 9007199254740991

def sha256_compact(data: dict) -> str:
    compact_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(compact_str.encode('utf-8')).hexdigest()


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
                    if not is_safe_int(rev, non_negative=True):
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

def is_safe_int(v, non_negative=False):
    if type(v) is not int: return False
    if non_negative and v < 0: return False
    return -9007199254740991 <= v <= 9007199254740991

def sha256_compact(data: dict) -> str:
    compact_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(compact_str.encode('utf-8')).hexdigest()

def handle_select(payload: dict):
    run_id, limit, forbidden, rows, trials = payload.get("runId"), payload.get("numTrialsLimit"), payload.get("forbiddenFeatures"), payload.get("rows"), payload.get("trials")
    
    invalid = False
    if type(run_id) is not str or not run_id or len(run_id) > 128: invalid = True
    if not is_safe_int(limit) or limit <= 0: invalid = True
    if type(forbidden) is not list or not all(type(x) is str for x in forbidden): invalid = True
    if type(rows) is not list or not rows or type(trials) is not list: invalid = True
    
    if invalid: return {"runId": payload.get("runId"), "selectedTrialId": None, "trainRowIds": [], "evalRowIds": [], "featureNames": [], "datasetDigest": None, "reasonCodes": ["INVALID_INPUT"]}
        
    if run_id in _STATE: return _STATE[run_id]["response"] if _STATE[run_id]["request"] == payload else {"_conflict": True}
            
    row_ids, groups = set(), {}
    for r in rows:
        if type(r) is not dict: invalid = True; break
        rid = r.get("id")
        if type(rid) is not str or rid in row_ids: invalid = True; break
        row_ids.add(rid)
        
        ent, evt, pt, ver, split, feats = r.get("entity"), r.get("eventTime"), r.get("predictionTime"), r.get("version"), r.get("split"), r.get("features")
        if type(ent) is not str or type(evt) is not str or type(pt) is not str or type(split) is not str: invalid = True; break
        if split not in ("TRAIN", "EVAL") or not is_safe_int(ver, non_negative=True) or type(feats) is not dict: invalid = True; break
        
        evt_dt, pt_dt = parse_and_validate_time(evt), parse_and_validate_time(pt)
        if not evt_dt or not pt_dt: invalid = True; break
        
        for fn, fv in feats.items():
            if type(fn) is not str or type(fv) is not dict or "value" not in fv: invalid = True; break
            avail = fv.get("availableAt")
            if type(avail) is not str or not parse_and_validate_time(avail): invalid = True; break
            
        if invalid: break
        
        r.update({"_evt_dt": evt_dt, "_pt_dt": pt_dt, "_id_bytes": rid.encode('utf-8')})
        key = (ent, evt_dt.timestamp())
        if key not in groups: groups[key] = []
        groups[key].append(r)
        
    if invalid: return {"runId": payload.get("runId"), "selectedTrialId": None, "trainRowIds": [], "evalRowIds": [], "featureNames": [], "datasetDigest": None, "reasonCodes": ["INVALID_INPUT"]}
        
    retained = [sorted(members, key=lambda x: (-x["version"], x["_id_bytes"]))[0] for members in groups.values()]
        
    f_counts, f_valid = {}, {}
    for r in retained:
        for fn, fv in r["features"].items():
            if fn in forbidden: continue
            avail_dt = parse_and_validate_time(fv["availableAt"])
            f_counts[fn] = f_counts.get(fn, 0) + 1
            if fn not in f_valid: f_valid[fn] = True
            if avail_dt > r["_pt_dt"]: f_valid[fn] = False
            
    v_feats = sorted([fn for fn, count in f_counts.items() if count == len(retained) and f_valid[fn]], key=lambda x: x.encode('utf-8'))
    t_ids = sorted([r["id"] for r in retained if r["split"] == "TRAIN"], key=lambda x: x.encode('utf-8'))
    e_ids = sorted([r["id"] for r in retained if r["split"] == "EVAL"], key=lambda x: x.encode('utf-8'))
    digest = sha256_compact({"trainRowIds": t_ids, "evalRowIds": e_ids, "featureNames": v_feats})
    
    trial_ids, best_trial = set(), None
    for t in trials:
        if type(t) is not dict: invalid = True; break
        tid, status, metric = t.get("trialId"), t.get("status"), t.get("evalMetric")
        if not is_safe_int(tid, non_negative=True) or tid in trial_ids or status not in ("SUCCEEDED", "FAILED"): invalid = True; break
        trial_ids.add(tid)
        
        if "evalMetric" in t:
            if type(metric) not in (float, int) or not math.isfinite(metric): invalid = True; break
        elif status == "SUCCEEDED": invalid = True; break
            
        if status == "SUCCEEDED" and (best_trial is None or metric > best_trial["metric"] or (metric == best_trial["metric"] and tid < best_trial["id"])):
            best_trial = {"id": tid, "metric": metric}
                
    if invalid: return {"runId": payload.get("runId"), "selectedTrialId": None, "trainRowIds": [], "evalRowIds": [], "featureNames": [], "datasetDigest": None, "reasonCodes": ["INVALID_INPUT"]}
        
    codes = []
    if len(trials) > limit: codes.append("TRIAL_LIMIT_EXCEEDED")
    elif not best_trial: codes.append("NO_SUCCESSFUL_TRIAL")
        
    res = {
        "runId": run_id, "selectedTrialId": best_trial["id"] if not codes else None,
        "trainRowIds": t_ids, "evalRowIds": e_ids, "featureNames": v_feats,
        "datasetDigest": digest, "reasonCodes": codes
    }
    if not codes: _STATE[run_id] = {"request": payload, "response": res}
    return res

def handle_evaluate(payload: dict):
    run_id, trial_id, digest = payload.get("runId"), payload.get("selectedTrialId"), payload.get("datasetDigest")
    floor_agg, req_slices, bp, mb, rows = payload.get("metricFloor"), payload.get("requiredSlices"), payload.get("bytesProcessed"), payload.get("maxBytes"), payload.get("rows")
    
    codes, critical = set(), True
    
    # 1. Structural Validation (No Hard-Abort, Aggregate All Violations)
    if type(run_id) is not str or not run_id or len(run_id) > 128: codes.add("INVALID_INPUT")
    if type(trial_id) is not int or trial_id < 0: codes.add("INVALID_INPUT")
    if type(digest) is not str or not re.fullmatch(r'^[0-9a-f]{64}$', digest): codes.add("INVALID_INPUT")
    if type(floor_agg) not in (float, int) or not (0 <= floor_agg <= 1) or not math.isfinite(floor_agg): codes.add("INVALID_INPUT")
    if type(req_slices) is not dict or not all(type(v) in (float, int) and 0 <= v <= 1 and math.isfinite(v) for v in req_slices.values()): codes.add("INVALID_INPUT")
    if not is_safe_int(bp, non_negative=True) or not is_safe_int(mb, non_negative=True): codes.add("INVALID_INPUT")
    if type(rows) is not list: codes.add("INVALID_INPUT")

    if "INVALID_INPUT" in codes: critical = False

    # 2. Lineage Integrity Validation
    if type(run_id) is str and type(trial_id) is int and type(digest) is str and re.fullmatch(r'^[0-9a-f]{64}$', digest):
        if run_id not in _STATE or _STATE[run_id]["response"]["selectedTrialId"] != trial_id or _STATE[run_id]["response"]["datasetDigest"] != digest:
            codes.add("INVALID_LINEAGE"); critical = False
    else:
        codes.add("INVALID_LINEAGE"); critical = False

    # 3. Byte Overrun Assessment
    if type(bp) is int and type(mb) is int and bp >= 0 and mb >= 0:
        if bp > mb: codes.add("BYTE_LIMIT")

    # 4. Matrix Profiling & Row Validation
    valid_rows, stats, c_tot, t_tot = True, {}, 0, 0
    if type(rows) is list:
        if not rows:
            valid_rows, critical = False, False
        else:
            for r in rows:
                if type(r) is not dict:
                    codes.add("INVALID_TEST_ROW"); valid_rows = False; continue
                lbl, prd, slc = r.get("label"), r.get("prediction"), r.get("slice")
                
                if type(lbl) is not int or lbl not in (0, 1) or type(prd) is not int or prd not in (0, 1) or type(slc) is not str or not slc:
                    codes.add("INVALID_TEST_ROW"); valid_rows = False; continue
                    
                t_tot += 1
                if lbl == prd: c_tot += 1
                if slc not in stats: stats[slc] = {"c": 0, "t": 0}
                stats[slc]["t"] += 1
                if lbl == prd: stats[slc]["c"] += 1
    else:
        valid_rows, critical = False, False
        
    tm = None
    if not valid_rows:
        critical = False
    else:
        tm = round(c_tot / t_tot, 12)
        if type(floor_agg) in (int, float) and tm < floor_agg: 
            codes.add("AGGREGATE_FLOOR")
            
        if type(req_slices) is dict:
            for rs, rsf in req_slices.items():
                if rs not in stats: 
                    codes.add(f"MISSING_SLICE:{rs}"); critical = False
                elif type(rsf) in (int, float):
                    if round(stats[rs]["c"] / stats[rs]["t"], 12) < rsf:
                        codes.add(f"SLICE_FLOOR:{rs}"); critical = False
                        
    dec = "admit" if not codes and tm is not None and critical else "reject"
    
    return {
        "runId": payload.get("runId"), "selectedTrialId": payload.get("selectedTrialId"), "datasetDigest": payload.get("datasetDigest"),
        "testMetric": tm, "criticalSlicePass": critical, "decision": dec, "bytesProcessed": payload.get("bytesProcessed"),
        "reasonCodes": sorted(list(codes), key=lambda x: x.encode('utf-8'))
    }

@app.post("/bqml")
async def bqml_endpoint(request: Request):
    try: payload = json.loads(await request.body())
    except Exception: return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")
        
    if type(payload) is not dict: return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    phase = payload.get("phase")
    if phase == "select":
        res = handle_select(payload)
        if res.get("_conflict"): return Response(content=json.dumps({"error": "RUN_ID_CONFLICT"}), status_code=409, media_type="application/json")
        return Response(content=json.dumps(res, ensure_ascii=False, separators=(',', ':')), status_code=200, media_type="application/json")
    
    elif phase == "evaluate":
        return Response(content=json.dumps(handle_evaluate(payload), ensure_ascii=False, separators=(',', ':')), status_code=200, media_type="application/json")
    
    else:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")


# ==========================================
# Q3: MLFLOW MODEL PROMOTION
# ==========================================
def is_canonical_positive_safe_int(v_str: str) -> bool:
    if type(v_str) is not str: return False
    if not re.fullmatch(r'^[1-9][0-9]*$', v_str): return False
    return int(v_str) <= 9007199254740991

@app.post("/promote")
@app.post("/promote/")
async def promote_endpoint(request: Request):
    try: payload = json.loads(await request.body())
    except Exception: return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")
        
    if type(payload) is not dict: return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    as_of_str = payload.get("asOf")
    champ_v = payload.get("championVersion")
    policy = payload.get("policy")
    versions = payload.get("versions")
    
    if type(policy) is not dict or type(versions) is not list or type(champ_v) is not str:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    as_of_dt = parse_and_validate_time(as_of_str)
    if not as_of_dt: return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    p_dataset = policy.get("datasetDigest")
    p_schema = policy.get("schemaDigest")
    p_age = policy.get("maxAgeSeconds")
    p_acc = policy.get("accuracyFloor")
    p_slices = policy.get("requiredSlices")
    p_lat = policy.get("maxLatencyMs")
    p_size = policy.get("maxSizeBytes")
    p_imp = policy.get("minImprovement")

    # Policy Validation
    p_invalid = False
    if type(p_dataset) is not str or not p_dataset or type(p_schema) is not str or not p_schema: p_invalid = True
    if not is_safe_int(p_age, non_negative=True) or not is_safe_int(p_size, non_negative=True): p_invalid = True
    if type(p_acc) not in (float, int) or not (0 <= p_acc <= 1) or not math.isfinite(p_acc): p_invalid = True
    if type(p_lat) not in (float, int) or p_lat < 0 or not math.isfinite(p_lat): p_invalid = True
    if type(p_imp) not in (float, int) or not (0 <= p_imp <= 1) or not math.isfinite(p_imp): p_invalid = True
    if type(p_slices) is not dict or not all(type(v) in (float, int) and 0 <= v <= 1 and math.isfinite(v) for v in p_slices.values()): p_invalid = True

    # Pre-scan for duplicates
    v_counts = {}
    for v_obj in versions:
        if type(v_obj) is dict and type(v_obj.get("version")) is str:
            v_str = v_obj["version"]
            v_counts[v_str] = v_counts.get(v_str, 0) + 1

    failed_gates = {}
    eligible_pool = []
    evidence_map = {}

    for v_obj in versions:
        if type(v_obj) is not dict: continue
        v_str = v_obj.get("version")
        v_art = v_obj.get("artifactDigest")
        v_eval = v_obj.get("evaluation")
        
        if type(v_str) is not str: continue
        
        codes = set()
        
        if not is_canonical_positive_safe_int(v_str): codes.add("INVALID_VERSION")
        elif v_counts.get(v_str, 0) > 1: codes.add("DUPLICATE_VERSION")
        
        if p_invalid: codes.add("INVALID_POLICY")

        if type(v_eval) is not dict:
            codes.add("MISSING_EVALUATION")
        else:
            e_ca = v_eval.get("createdAt")
            e_art = v_eval.get("artifactDigest")
            e_dat = v_eval.get("datasetDigest")
            e_sch = v_eval.get("schemaDigest")
            e_acc = v_eval.get("accuracy")
            e_lat = v_eval.get("latencyMs")
            e_size = v_eval.get("sizeBytes")
            e_slices = v_eval.get("slices")

            # Timestamps
            e_dt = parse_and_validate_time(e_ca)
            if not e_dt:
                codes.add("INVALID_TIMESTAMP")
            else:
                if e_dt > as_of_dt: codes.add("FUTURE_EVALUATION")
                elif type(p_age) is int and e_dt < (as_of_dt - timedelta(seconds=p_age)): codes.add("STALE_EVALUATION")

            # Artifact/Digests
            if e_art != v_art: codes.add("ARTIFACT_MISMATCH")
            if type(p_dataset) is str and e_dat != p_dataset: codes.add("DATASET_MISMATCH")
            if type(p_schema) is str and e_sch != p_schema: codes.add("SCHEMA_MISMATCH")

            # Metrics Formats
            if type(e_acc) not in (float, int) or type(e_lat) not in (float, int) or not is_safe_int(e_size, non_negative=True):
                codes.add("NON_FINITE")
            else:
                if not math.isfinite(e_acc) or not math.isfinite(e_lat): codes.add("NON_FINITE")
                else:
                    if not (0 <= e_acc <= 1): codes.add("METRIC_RANGE")
                    elif type(p_acc) in (float, int) and e_acc < p_acc: codes.add("ACCURACY_FLOOR")
                    
                    if e_lat < 0: codes.add("METRIC_RANGE")
                    elif type(p_lat) in (float, int) and e_lat > p_lat: codes.add("LATENCY_LIMIT")

                    if type(p_size) is int and e_size > p_size: codes.add("SIZE_LIMIT")

            # Slices
            if type(e_slices) is not dict:
                if type(p_slices) is dict and p_slices:
                    for req_slc in p_slices: codes.add(f"MISSING_SLICE:{req_slc}")
            else:
                if type(p_slices) is dict:
                    for req_slc, slc_floor in p_slices.items():
                        if req_slc not in e_slices:
                            codes.add(f"MISSING_SLICE:{req_slc}")
                        else:
                            slc_val = e_slices[req_slc]
                            if type(slc_val) not in (float, int) or not math.isfinite(slc_val):
                                codes.add(f"SLICE_RANGE:{req_slc}")
                            elif not (0 <= slc_val <= 1):
                                codes.add(f"SLICE_RANGE:{req_slc}")
                            elif slc_val < slc_floor:
                                codes.add(f"SLICE_FLOOR:{req_slc}")

        if codes:
            failed_gates[v_str] = sorted(list(codes), key=lambda x: x.encode('utf-8'))
        else:
            evidence_map[v_str] = v_eval
            eligible_pool.append({
                "v_str": v_str,
                "v_int": int(v_str),
                "acc": v_eval["accuracy"],
                "lat": v_eval["latencyMs"],
                "size": v_eval["sizeBytes"]
            })

    # Ranking: Accuracy (desc), Latency (asc), Size (asc), Version Int (asc)
    eligible_pool.sort(key=lambda x: (-x["acc"], x["lat"], x["size"], x["v_int"]))
    eligible_versions = [x["v_str"] for x in eligible_pool]

    action = "retain"
    selected = champ_v
    mutation = None
    final_evidence = None

    if champ_v in failed_gates or champ_v not in evidence_map:
        action = "block"
        selected = None
    elif eligible_pool:
        challenger = eligible_pool[0]
        c_str = challenger["v_str"]
        c_acc = challenger["acc"]
        champ_acc = evidence_map[champ_v]["accuracy"]
        
        diff = round(c_acc - champ_acc, 12)
        if type(p_imp) in (float, int) and diff >= p_imp:
            if c_str != champ_v:
                action = "promote"
                selected = c_str
                mutation = {"alias": "champion", "version": c_str}
        
    if selected and selected in evidence_map:
        final_evidence = evidence_map[selected]

    res = {
        "action": action,
        "championVersion": champ_v,
        "selectedVersion": selected,
        "eligibleVersions": eligible_versions,
        "failedGates": failed_gates,
        "aliasMutation": mutation,
        "evidence": final_evidence
    }

    return Response(content=json.dumps(res, ensure_ascii=False, separators=(',', ':')), status_code=200, media_type="application/json")