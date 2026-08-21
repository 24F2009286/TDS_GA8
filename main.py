from fastapi import FastAPI, Request, Response
import json
import re
import math
from datetime import datetime, timezone, timedelta

app = FastAPI()

# In-memory state store: runId -> {"request": original_payload, "response": response_payload}
_STATE = {}

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

import hashlib

def sha256_compact(data: dict) -> str:
    compact_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(compact_str.encode('utf-8')).hexdigest()

def handle_select(payload: dict):
    run_id = payload.get("runId")
    if type(run_id) is not str or not run_id or len(run_id) > 128:
        return None, ["INVALID_INPUT"]
    
    # Check idempotency / conflicts
    if run_id in _STATE:
        if _STATE[run_id]["request"] == payload:
            return _STATE[run_id]["response"], None
        else:
            return "CONFLICT", None

    limit = payload.get("numTrialsLimit")
    if type(limit) is not int or limit <= 0: return None, ["INVALID_INPUT"]

    rows = payload.get("rows")
    trials = payload.get("trials")
    if type(rows) is not list or not rows or type(trials) is not list:
        return None, ["INVALID_INPUT"]

    # 1. Deduplicate Rows
    groups = {}
    for r in rows:
        if type(r) is not dict: return None, ["INVALID_INPUT"]
        r_id = r.get("id")
        ent = r.get("entity")
        evt = r.get("eventTime")
        pt = r.get("predictionTime")
        ver = r.get("version")
        split = r.get("split")
        feats = r.get("features")
        
        if not all(type(x) is str for x in [r_id, ent, evt, pt, split]): return None, ["INVALID_INPUT"]
        if type(ver) is not int or ver < 0: return None, ["INVALID_INPUT"]
        if split not in ("TRAIN", "EVAL"): return None, ["INVALID_INPUT"]
        if type(feats) is not dict: return None, ["INVALID_INPUT"]
        
        evt_dt = parse_and_validate_time(evt)
        pt_dt = parse_and_validate_time(pt)
        if not evt_dt or not pt_dt: return None, ["INVALID_INPUT"]
        
        r["_evt_dt"] = evt_dt
        r["_pt_dt"] = pt_dt
        r["_id_bytes"] = r_id.encode('utf-8')
        
        key = (ent, evt_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
        if key not in groups: groups[key] = []
        groups[key].append(r)

    retained = []
    for key, members in groups.items():
        members.sort(key=lambda x: (-x["version"], x["_id_bytes"]))
        retained.append(members[0])

    if not retained: return None, ["INVALID_INPUT"]

    # 2. Feature Eligibility
    forbidden = set(payload.get("forbiddenFeatures", []))
    feature_counts = {}
    feature_valid = {}
    
    for r in retained:
        for fname, fval in r["features"].items():
            if fname in forbidden: continue
            if type(fval) is not dict: return None, ["INVALID_INPUT"]
            avail_dt = parse_and_validate_time(fval.get("availableAt", ""))
            if not avail_dt: return None, ["INVALID_INPUT"]
            
            feature_counts[fname] = feature_counts.get(fname, 0) + 1
            if fname not in feature_valid: feature_valid[fname] = True
            if avail_dt > r["_pt_dt"]: feature_valid[fname] = False

    valid_features = []
    req_count = len(retained)
    for fname, count in feature_counts.items():
        if count == req_count and feature_valid[fname]:
            valid_features.append(fname)
            
    valid_features.sort(key=lambda x: x.encode('utf-8'))

    # 3. Trial Selection
    if len(trials) > limit:
        return None, ["TRIAL_LIMIT_EXCEEDED"]
        
    best_trial = None
    for t in trials:
        if type(t) is not dict: return None, ["INVALID_INPUT"]
        t_id = t.get("trialId")
        status = t.get("status")
        metric = t.get("evalMetric")
        
        if type(t_id) is not int or t_id < 0: return None, ["INVALID_INPUT"]
        if status not in ("SUCCEEDED", "FAILED"): return None, ["INVALID_INPUT"]
        if type(metric) not in (float, int) or not math.isfinite(metric): return None, ["INVALID_INPUT"]
        
        if status == "SUCCEEDED":
            if best_trial is None:
                best_trial = {"id": t_id, "metric": metric}
            else:
                if metric > best_trial["metric"]:
                    best_trial = {"id": t_id, "metric": metric}
                elif metric == best_trial["metric"] and t_id < best_trial["id"]:
                    best_trial = {"id": t_id, "metric": metric}

    if not best_trial:
        return None, ["NO_SUCCESSFUL_TRIAL"]

    # 4. Generate Output
    train_ids = [r["id"] for r in retained if r["split"] == "TRAIN"]
    eval_ids = [r["id"] for r in retained if r["split"] == "EVAL"]
    train_ids.sort(key=lambda x: x.encode('utf-8'))
    eval_ids.sort(key=lambda x: x.encode('utf-8'))

    digest_payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": valid_features
    }
    
    response = {
        "runId": run_id,
        "selectedTrialId": best_trial["id"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": valid_features,
        "datasetDigest": sha256_compact(digest_payload),
        "reasonCodes": []
    }
    
    _STATE[run_id] = {"request": payload, "response": response}
    return response, None

def handle_evaluate(payload: dict):
    run_id = payload.get("runId")
    trial_id = payload.get("selectedTrialId")
    digest = payload.get("datasetDigest")
    
    if type(run_id) is not str or type(trial_id) is not int or type(digest) is not str:
        return None, ["INVALID_INPUT"]
        
    codes = set()
    test_metric = None
    critical_pass = True
    
    # Lineage check
    if run_id not in _STATE:
        codes.add("INVALID_LINEAGE")
        critical_pass = False
    else:
        saved = _STATE[run_id]["response"]
        if saved["selectedTrialId"] != trial_id or saved["datasetDigest"] != digest:
            codes.add("INVALID_LINEAGE")
            critical_pass = False

    floor_agg = payload.get("metricFloor")
    req_slices = payload.get("requiredSlices")
    bp = payload.get("bytesProcessed")
    mb = payload.get("maxBytes")
    rows = payload.get("rows")
    
    if type(floor_agg) not in (float, int) or not (0 <= floor_agg <= 1): codes.add("INVALID_INPUT")
    if type(req_slices) is not dict: codes.add("INVALID_INPUT")
    if type(bp) is not int or bp < 0 or type(mb) is not int or mb < 0: codes.add("INVALID_INPUT")
    if type(rows) is not list: codes.add("INVALID_INPUT")

    if "INVALID_INPUT" in codes:
        critical_pass = False
        return None, list(codes), critical_pass

    # Byte check
    if bp > mb:
        codes.add("BYTE_LIMIT")

    valid_rows = True
    if not rows:
        valid_rows = False
        critical_pass = False
        
    slice_stats = {}
    total_correct = 0
    total_rows = 0
    
    for r in rows:
        if type(r) is not dict:
            codes.add("INVALID_TEST_ROW")
            valid_rows = False
            continue
            
        lbl = r.get("label")
        prd = r.get("prediction")
        slc = r.get("slice")
        
        if lbl not in (0, 1) or prd not in (0, 1) or type(slc) is not str or not slc:
            codes.add("INVALID_TEST_ROW")
            valid_rows = False
            continue
            
        total_rows += 1
        is_correct = (lbl == prd)
        if is_correct: total_correct += 1
        
        if slc not in slice_stats: slice_stats[slc] = {"c": 0, "t": 0}
        slice_stats[slc]["t"] += 1
        if is_correct: slice_stats[slc]["c"] += 1

    if not valid_rows:
        critical_pass = False
    else:
        # Aggregates
        test_metric = round(total_correct / total_rows, 12)
        if test_metric < floor_agg:
            codes.add("AGGREGATE_FLOOR")
            
        for req_slice, s_floor in req_slices.items():
            if type(s_floor) not in (float, int) or not (0 <= s_floor <= 1):
                codes.add("INVALID_INPUT")
                critical_pass = False
                continue
                
            if req_slice not in slice_stats:
                codes.add(f"MISSING_SLICE:{req_slice}")
                critical_pass = False
            else:
                s_acc = round(slice_stats[req_slice]["c"] / slice_stats[req_slice]["t"], 12)
                if s_acc < s_floor:
                    codes.add(f"SLICE_FLOOR:{req_slice}")
                    critical_pass = False

    decision = "admit" if (not codes and test_metric is not None and critical_pass) else "reject"
    
    # Enforce formatting constraints
    if codes:
        codes_list = sorted(list(codes), key=lambda x: x.encode('utf-8'))
    else:
        codes_list = []
        
    response = {
        "runId": run_id,
        "selectedTrialId": trial_id,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_pass,
        "decision": decision,
        "bytesProcessed": bp,
        "reasonCodes": codes_list
    }
    
    return response, None, None

@app.post("/bqml")
async def bqml_endpoint(request: Request):
    try:
        body_bytes = await request.body()
        payload = json.loads(body_bytes)
    except Exception:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")
        
    if type(payload) is not dict:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")

    phase = payload.get("phase")
    
    if phase == "select":
        res, errs = handle_select(payload)
        if errs:
            if errs[0] == "CONFLICT":
                return Response(content=json.dumps({"error": "RUN_ID_CONFLICT"}), status_code=409, media_type="application/json")
            
            # Error shaping
            err_res = {
                "runId": payload.get("runId") if type(payload.get("runId")) is str else "",
                "selectedTrialId": None,
                "trainRowIds": [],
                "evalRowIds": [],
                "featureNames": [],
                "datasetDigest": None,
                "reasonCodes": sorted(errs, key=lambda x: x.encode('utf-8'))
            }
            return Response(content=json.dumps(err_res), status_code=200, media_type="application/json")
            
        return Response(content=json.dumps(res), status_code=200, media_type="application/json")

    elif phase == "evaluate":
        res, errs, _ = handle_evaluate(payload)
        if errs and res is None:
            # Fatal input error mapping
            err_res = {
                "runId": payload.get("runId") if type(payload.get("runId")) is str else "",
                "selectedTrialId": payload.get("selectedTrialId") if type(payload.get("selectedTrialId")) is int else 0,
                "datasetDigest": payload.get("datasetDigest") if type(payload.get("datasetDigest")) is str else "",
                "testMetric": None,
                "criticalSlicePass": False,
                "decision": "reject",
                "bytesProcessed": payload.get("bytesProcessed") if type(payload.get("bytesProcessed")) is int else 0,
                "reasonCodes": sorted(errs, key=lambda x: x.encode('utf-8'))
            }
            return Response(content=json.dumps(err_res), status_code=200, media_type="application/json")
            
        return Response(content=json.dumps(res), status_code=200, media_type="application/json")
        
    else:
        return Response(content=json.dumps({"error": "INVALID_INPUT"}), status_code=400, media_type="application/json")