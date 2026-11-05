"""Improved v2 - Advanced Features, Four-Model Blend and Calibration (42 dimensions).

Reconstructed from Section 5 of the project report:
  * same imputation / S-M-H encoding as v1
  * 10 raw sensors + unit_type_enc + 17 domain/flag features + 14 unit-relative features = 42
  * blend: HGB (0.35) + HGB (0.25) + RandomForest (0.20) + ExtraTrees (0.20)
  * StratifiedKFold(10, shuffle=True, random_state=42)
  * class-wise probability multipliers tuned with Nelder-Mead (start [1,1,1],
    maxiter 500, normalised by the max element)

NOTE: the report lists the NAMES of the 17 domain features but not their formulas
(only the four flag thresholds are given). The formulas below are reasonable
reconstructions - replace them with the exact ones from your original notebook
if they differ.

Usage:
    python improved_v2.py --data-dir . --out-dir .
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

TARGET = "condition"
ID_COL = "unit_id"
NUM_COLS = [
    "ambient_temp", "process_temp", "coolant_flow", "shaft_speed", "drive_torque",
    "power_load", "component_wear", "thermal_stress", "mechanical_load", "vibration_index",
]
UNIT_MAP = {"S": 0, "M": 1, "H": 2}
UNIT_REL_COLS = ["process_temp", "shaft_speed", "drive_torque", "power_load",
                 "thermal_stress", "mechanical_load", "component_wear"]
WEIGHTS = [0.35, 0.25, 0.20, 0.20]
N_SPLITS = 10
SEED = 42
EPS = 1e-5


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def clean_unit_type(series):
    return series.astype(str).str.strip().str.upper()


def impute(df, stats):
    x = pd.DataFrame(index=df.index)
    for c in NUM_COLS:
        x[c] = pd.to_numeric(df[c], errors="coerce").fillna(stats["medians"][c])
    ut = clean_unit_type(df["unit_type"])
    ut = ut.where(ut.isin(UNIT_MAP), stats["unit_mode"])
    x["unit_type_enc"] = ut.map(UNIT_MAP).astype(int)
    return x


def fit_stats(train):
    medians = {c: pd.to_numeric(train[c], errors="coerce").median() for c in NUM_COLS}
    ut = clean_unit_type(train["unit_type"])
    stats = {"medians": medians, "unit_mode": ut[ut.isin(UNIT_MAP)].mode().iloc[0]}
    # unit-type-specific mean / std (on imputed training data) for the 7 variables
    imputed = impute(train, stats)
    g = imputed.groupby("unit_type_enc")[UNIT_REL_COLS]
    stats["group_mean"] = g.mean()
    stats["group_std"] = g.std(ddof=0)
    return stats


def build_features(df, stats):
    x = impute(df, stats)

    # --- Thermal / cooling (3)
    x["delta_temp"] = x["process_temp"] - x["ambient_temp"]
    x["cooling_efficiency"] = x["coolant_flow"] / (x["delta_temp"].abs() + EPS)
    x["thermal_to_coolant"] = x["thermal_stress"] / (x["coolant_flow"] + EPS)

    # --- Power / mechanical (4)
    x["theoretical_power"] = x["shaft_speed"] * x["drive_torque"] / 9550.0
    x["power_discrepancy"] = x["power_load"] - x["theoretical_power"]
    x["power_ratio"] = x["power_load"] / (x["theoretical_power"].abs() + EPS)
    x["overstrain_index"] = x["drive_torque"] * x["mechanical_load"]

    # --- Wear / vibration (4)
    x["wear_vibe"] = x["component_wear"] * x["vibration_index"]
    x["wear_vibe_sq"] = x["wear_vibe"] ** 2
    x["vibe_speed_energy"] = x["vibration_index"] * x["shaft_speed"]
    x["mechanical_vibe"] = x["mechanical_load"] * x["vibration_index"]

    # --- Stress (2)
    x["thermal_mechanical_ratio"] = x["thermal_stress"] / (x["mechanical_load"] + EPS)
    x["total_stress"] = x["thermal_stress"] + x["mechanical_load"]

    # --- Threshold flags (4)  (cut-offs from the report)
    x["flag_high_temp"] = (x["process_temp"] > 69.15).astype(int)
    x["flag_high_wear"] = (x["component_wear"] > 185.0).astype(int)
    x["flag_high_mech"] = (x["mechanical_load"] > 100.0).astype(int)
    x["flag_high_vibe"] = (x["vibration_index"] > 2.1).astype(int)

    # --- Unit-relative features (14): difference from unit-type mean and z-score
    unit = x["unit_type_enc"]
    for c in UNIT_REL_COLS:
        mu = unit.map(stats["group_mean"][c])
        sd = unit.map(stats["group_std"][c]) + 1e-6
        x[f"{c}_diff_unit"] = x[c] - mu
        x[f"{c}_z_unit"] = (x[c] - mu) / sd
    return x


# ----------------------------------------------------------------------------
# Models, blending, calibration
# ----------------------------------------------------------------------------
def make_models(fold):
    return [
        HistGradientBoostingClassifier(
            max_iter=450, learning_rate=0.035, max_leaf_nodes=42, min_samples_leaf=18,
            l2_regularization=0.8, random_state=42 + fold),
        HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.045, max_leaf_nodes=31, min_samples_leaf=25,
            l2_regularization=1.5, random_state=100 + fold),
        RandomForestClassifier(
            n_estimators=300, max_depth=17, min_samples_leaf=2, max_features="sqrt",
            random_state=42 + fold, n_jobs=-1),
        ExtraTreesClassifier(
            n_estimators=300, max_depth=19, min_samples_leaf=2,
            random_state=77 + fold, n_jobs=-1),
    ]


def blend_proba(models, X):
    return sum(w * m.predict_proba(X) for w, m in zip(WEIGHTS, models))


def fit_class_multipliers(proba, y):
    """Nelder-Mead search for per-class multipliers that maximise accuracy."""
    def neg_acc(w):
        return -accuracy_score(y, (proba * w).argmax(1))

    res = minimize(neg_acc, x0=np.ones(proba.shape[1]), method="Nelder-Mead",
                   options={"maxiter": 500})
    w = np.abs(res.x)          # keep multipliers positive
    return w / w.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--train", default="student_dataset(1).csv")
    ap.add_argument("--test", default="test(4).csv")
    ap.add_argument("--sample", default="sample_submission(1).csv")
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(args.data_dir, args.train))
    test = pd.read_csv(os.path.join(args.data_dir, args.test))
    sample = pd.read_csv(os.path.join(args.data_dir, args.sample))

    y = train[TARGET].astype(int).to_numpy()
    stats = fit_stats(train)
    X = build_features(train, stats)
    X_test = build_features(test, stats)
    assert X.shape[1] == 42, f"expected 42 features, got {X.shape[1]}"
    print(f"Train matrix {X.shape}, test matrix {X_test.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3))
    test_proba = np.zeros((len(X_test), 3))

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        models = make_models(fold)
        for m in models:
            m.fit(X.iloc[tr], y[tr])
        oof[va] = blend_proba(models, X.iloc[va])
        test_proba += blend_proba(models, X_test) / N_SPLITS
        print(f"Fold {fold + 1:2d}: accuracy = {accuracy_score(y[va], oof[va].argmax(1)):.5f}")

    raw_pred = oof.argmax(1)
    print(f"\nRaw 10-fold OOF accuracy       : {accuracy_score(y, raw_pred):.5f}")

    mult = fit_class_multipliers(oof, y)
    cal_pred = (oof * mult).argmax(1)
    print(f"Class multipliers              : {np.round(mult, 4)}")
    print(f"Calibrated OOF accuracy        : {accuracy_score(y, cal_pred):.5f}")
    print("(multipliers are tuned on the same OOF labels, so this is optimistic)")
    print(f"Calibrated OOF macro F1        : {f1_score(y, cal_pred, average='macro'):.4f}")
    print(classification_report(y, cal_pred, digits=4))

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: (test_proba * mult).argmax(1)})
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")
    out_path = os.path.join(args.out_dir, "submission_improved_v2.csv")
    sub.to_csv(out_path, index=False)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
