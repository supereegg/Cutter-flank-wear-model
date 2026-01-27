
# main.py 
# Runs inside the PHM portal container:
# - Reads eval data from /tcdata using the official DataLoader
# - Extracts features via lib.feature_engineering.get_features
# - Loads model trained on log1p(Δwear)
# - Applies np.expm1() to revert predictions to Δwear
# - Saves /work/result.csv with columns: set_num, cut_num, pred

import os, json, numpy as np, pandas as pd, joblib
from lib.data_loader import DataLoader
from lib.feature_engineering import get_features

MODEL_PATH = "model/final_model.joblib"
FEATURE_LIST = "model/feature_list.json"

CTRL_ROOT = "/tcdata/Controller_Data"
SENS_ROOT = "/tcdata/Sensor_Data"
OUT_PATH  = "/work/result.csv"

def log(m): print(m, flush=True)

def ensure_feature_order(feature_dicts, feature_list):
    """Convert list[dict] -> np.array aligned to feature_list; fill missing with 0.0"""
    X = np.zeros((len(feature_dicts), len(feature_list)), dtype=np.float32)
    for i, fd in enumerate(feature_dicts):
        for j, name in enumerate(feature_list):
            val = fd.get(name, 0.0)
            try:
                X[i, j] = np.float32(val)
            except Exception:
                X[i, j] = 0.0
    return X

def main():
    # Load model and feature list
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError("Model file not found at " + MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        feat_names_model = bundle.get("feature_names", None)
    else:
        model = bundle
        feat_names_model = None

    # Prefer JSON feature list if present
    if os.path.isfile(FEATURE_LIST):
        with open(FEATURE_LIST, "r") as f:
            feat_names = json.load(f)
    else:
        feat_names = feat_names_model if feat_names_model is not None else []
    if not feat_names:
        raise RuntimeError("No feature list available to align inputs. Include model/feature_list_delta.json.")

    loader = DataLoader(CTRL_ROOT, SENS_ROOT)

    evalset_list = [1, 2, 3]
    cut_list = list(range(2, 27))

    records = []
    feats_all = []

    for set_no in evalset_list:
        set_name = f"evalset_{set_no:02d}"
        for cut_no in cut_list:
            log(f"[EVAL] {set_name} Cut {cut_no} ...")
            ctrl_df = loader.get_controller_data(set_no, cut_no)
            sens_df = loader.get_sensor_data(set_no, cut_no)
            feats = get_features(ctrl_df, sens_df)
            feats_all.append(feats)
            records.append([set_name, cut_no])

    # Align features and predict
    X = ensure_feature_order(feats_all, feat_names)
    # y_pred_log = model.predict(X).astype(np.float32)
    
    y_pred = model.predict(X).astype(np.float32)
    
    # # Inverse transform: log1p(Δwear) -> Δwear
    # y_pred = np.expm1(y_pred_log)
    # y_pred = np.where(y_pred < 0, 0.0, y_pred)

    # Save predictions
    out = pd.DataFrame(records, columns=["set_num","cut_num"])
    out["pred"] = y_pred
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    log(f"[DONE] Saved predictions to {OUT_PATH}")

if __name__ == "__main__":
    main()
