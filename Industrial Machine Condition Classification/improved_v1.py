"""Improved v1 - Domain Features and Soft Voting (15 input dimensions).

Reconstructed from Section 4 of the project report:
  * median imputation for the 10 sensor columns, training-mode for unit_type
  * unit_type encoded S=0, M=1, H=2
  * 4 engineered features -> 15 columns total
  * probability blend: HGB (0.45) + HGB (0.35) + RandomForest (0.20)
  * StratifiedKFold(5, shuffle=True, random_state=42), aggregate OOF metrics

Usage:
    python improved_v1.py --data-dir . --out-dir .
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

TARGET = "condition"
ID_COL = "unit_id"
NUM_COLS = [
    "ambient_temp", "process_temp", "coolant_flow", "shaft_speed", "drive_torque",
    "power_load", "component_wear", "thermal_stress", "mechanical_load", "vibration_index",
]
UNIT_MAP = {"S": 0, "M": 1, "H": 2}
WEIGHTS = [0.45, 0.35, 0.20]
N_SPLITS = 5
SEED = 42


# ----------------------------------------------------------------------------
# Preprocessing (statistics come from the training set and are reused on test)
# ----------------------------------------------------------------------------
def clean_unit_type(series):
    return series.astype(str).str.strip().str.upper()


def fit_stats(train):
    medians = {c: pd.to_numeric(train[c], errors="coerce").median() for c in NUM_COLS}
    ut = clean_unit_type(train["unit_type"])
    ut = ut[ut.isin(UNIT_MAP)]
    return {"medians": medians, "unit_mode": ut.mode().iloc[0]}


def build_features(df, stats):
    x = pd.DataFrame(index=df.index)
    # empty / "nan" / unparseable numeric values -> training median
    for c in NUM_COLS:
        x[c] = pd.to_numeric(df[c], errors="coerce").fillna(stats["medians"][c])

    ut = clean_unit_type(df["unit_type"])
    ut = ut.where(ut.isin(UNIT_MAP), stats["unit_mode"])
    x["unit_type_enc"] = ut.map(UNIT_MAP).astype(int)

    # Engineered features are computed AFTER imputation
    x["temp_diff"] = x["process_temp"] - x["ambient_temp"]
    x["calculated_power"] = x["shaft_speed"] * x["drive_torque"] / 9550.0
    x["wear_vibe_ratio"] = x["component_wear"] * x["vibration_index"]  # product, as in report
    x["thermal_coolant_ratio"] = x["thermal_stress"] / (x["coolant_flow"] + 1e-5)
    return x


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_models(fold):
    return [
        HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.04, max_leaf_nodes=45,
            min_samples_leaf=20, random_state=42 + fold),
        HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=25,
            l2_regularization=1.0, random_state=100 + fold),
        RandomForestClassifier(
            n_estimators=250, max_depth=16, min_samples_leaf=3,
            random_state=42 + fold, n_jobs=-1),
    ]


def blend_proba(models, X):
    return sum(w * m.predict_proba(X) for w, m in zip(WEIGHTS, models))


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
    assert X.shape[1] == 15, f"expected 15 features, got {X.shape[1]}"
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
        acc = accuracy_score(y[va], oof[va].argmax(1))
        print(f"Fold {fold + 1}: accuracy = {acc:.5f}")

    pred = oof.argmax(1)
    print(f"\nOOF accuracy : {accuracy_score(y, pred):.4f}")
    print(f"OOF macro F1 : {f1_score(y, pred, average='macro'):.4f}")
    print(classification_report(y, pred, digits=4))

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_proba.argmax(1)})
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")
    out_path = os.path.join(args.out_dir, "submission_improved_v1.csv")
    sub.to_csv(out_path, index=False)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
