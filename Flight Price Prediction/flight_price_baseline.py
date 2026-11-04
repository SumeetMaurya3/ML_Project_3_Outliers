"""Flight-price regression with XGBoost and binned stratified K-fold validation.

Run:
    pip install pandas numpy scikit-learn xgboost
    python flight_price_xgboost_stratified.py

Input files are expected beside this script:
    train(3).csv, test(3).csv, submission(2).csv

Outputs:
    submission_xgboost.csv
    xgboost_cv_results.csv
    xgboost_oof_predictions.csv
    xgboost_validation_report.txt
"""
from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

BASE = Path(__file__).resolve().parent
TRAIN_PATH = BASE / "train(3).csv"
TEST_PATH = BASE / "test(3).csv"
SAMPLE_PATH = BASE / "submission(2).csv"
SEED = 42
N_SPLITS = 5
OUTLIER_PRICE_LIMIT = 5000.0  # removes only the two extremely implausible upper-tail records

# A small, reproducible randomized-style search spanning tree complexity,
# learning rate, row/feature subsampling and regularization.
PARAM_CANDIDATES = [
    dict(n_estimators=500, max_depth=3, learning_rate=0.03, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=5.0, gamma=0.0),
    dict(n_estimators=700, max_depth=4, learning_rate=0.03, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=5.0, gamma=0.0),
    dict(n_estimators=600, max_depth=5, learning_rate=0.03, min_child_weight=5, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=10.0, gamma=0.0),
    dict(n_estimators=900, max_depth=3, learning_rate=0.02, min_child_weight=3, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.00, reg_lambda=10.0, gamma=0.0),
    dict(n_estimators=700, max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.80, colsample_bytree=0.80, reg_alpha=0.10, reg_lambda=10.0, gamma=0.0),
    dict(n_estimators=500, max_depth=5, learning_rate=0.05, min_child_weight=5, subsample=0.85, colsample_bytree=0.90, reg_alpha=0.10, reg_lambda=5.0, gamma=0.0),
    dict(n_estimators=1000, max_depth=3, learning_rate=0.03, min_child_weight=1, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.00, reg_lambda=3.0, gamma=0.0),
    dict(n_estimators=800, max_depth=4, learning_rate=0.02, min_child_weight=3, subsample=0.80, colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=20.0, gamma=0.0),
    dict(n_estimators=600, max_depth=6, learning_rate=0.03, min_child_weight=8, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.50, reg_lambda=10.0, gamma=0.05),
    dict(n_estimators=900, max_depth=5, learning_rate=0.02, min_child_weight=5, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.00, reg_lambda=10.0, gamma=0.0),
    dict(n_estimators=700, max_depth=3, learning_rate=0.05, min_child_weight=1, subsample=0.80, colsample_bytree=0.90, reg_alpha=0.00, reg_lambda=5.0, gamma=0.0),
    dict(n_estimators=800, max_depth=4, learning_rate=0.03, min_child_weight=7, subsample=0.90, colsample_bytree=0.75, reg_alpha=0.50, reg_lambda=20.0, gamma=0.05),
]


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create robust features without using target or id as predictors."""
    x = df.copy()
    for col in ["airline", "flight", "source_city", "departure_time", "stops",
                "arrival_time", "destination_city", "flight_class"]:
        if col in x.columns:
            x[col] = clean_text(x[col]).fillna("Unknown")

    # Directional route and undirected city-pair features let the trees learn
    # route-specific fare patterns and share information between return routes.
    src = x["source_city"].astype(str)
    dst = x["destination_city"].astype(str)
    x["route"] = src + " -> " + dst
    x["city_pair"] = np.where(src <= dst, src + " <-> " + dst, dst + " <-> " + src)
    x["route_stops"] = x["route"] + " | " + x["stops"].astype(str)

    stop_map = {"zero": 0, "one": 1, "two_or_more": 2}
    x["stops_num"] = x["stops"].str.lower().map(stop_map).astype(float)

    # Binary class feature. Unknown classes safely map to NaN and are imputed.
    x["is_business"] = x["flight_class"].str.lower().map({"economy": 0.0, "business": 1.0})

    # Flight number is retained only as a weak numeric schedule identifier,
    # never one-hot encoded as a high-cardinality categorical field.
    x["flight_number"] = x["flight"].astype(str).str.extract(r"(\d+)", expand=False)
    x["flight_number"] = pd.to_numeric(x["flight_number"], errors="coerce")

    # Keep duration and days_left numeric and add a mild log transform for
    # smoother behavior near the lower end. Trees can ignore these features.
    x["duration_log1p"] = np.log1p(pd.to_numeric(x["duration"], errors="coerce").clip(lower=0))
    x["days_left_log1p"] = np.log1p(pd.to_numeric(x["days_left"], errors="coerce").clip(lower=0))

    # Drop identifiers and the original flight string (too many nearly-unique values).
    x = x.drop(columns=[c for c in ["id", "flight", "flight_class", "stops"] if c in x.columns])
    return x


def make_pipeline(x: pd.DataFrame, params: dict) -> Pipeline:
    categorical_cols = [c for c in x.columns if x[c].dtype == "object" or str(x[c].dtype).startswith("string")]
    numeric_cols = [c for c in x.columns if c not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)),
            ]), categorical_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        verbosity=0,
        **params,
    )
    return Pipeline([("preprocess", preprocessor), ("xgb", model)])


def main() -> None:
    started = time.time()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH) if SAMPLE_PATH.exists() else None

    if "price" not in train.columns:
        raise ValueError("Training file must contain a 'price' column.")
    original_n = len(train)
    train["price"] = pd.to_numeric(train["price"], errors="coerce")
    valid_price = train["price"].notna() & (train["price"] > 0) & (train["price"] <= OUTLIER_PRICE_LIMIT)
    removed_zero_or_invalid = int((train["price"].isna() | (train["price"] <= 0)).sum())
    removed_extreme = int((train["price"] > OUTLIER_PRICE_LIMIT).sum())
    train = train.loc[valid_price].reset_index(drop=True)

    y = train["price"].astype(float).to_numpy()
    X = make_features(train.drop(columns=["price"]))
    X_test = make_features(test)

    # Quantile bins of the cleaned target are used only to stratify folds.
    # qcut may merge bins when targets have ties; that's handled explicitly.
    target_bins = pd.qcut(pd.Series(y), q=10, labels=False, duplicates="drop")
    n_bins = int(pd.Series(target_bins).nunique())
    if n_bins < 2:
        raise ValueError("Target could not be split into multiple quantile bins.")
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, target_bins))

    all_results = []
    best_raw = None
    n_total = len(PARAM_CANDIDATES) + 1
    run_no = 0

    # Search the original-price target across all parameter candidates first.
    for cand_idx, params in enumerate(PARAM_CANDIDATES, start=1):
        run_no += 1
        oof = np.full(len(y), np.nan, dtype=float)
        fold_rmses, fold_maes = [], []
        for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
            model = make_pipeline(X.iloc[tr_idx], params)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            pred = np.clip(model.predict(X.iloc[va_idx]), 0, None)
            oof[va_idx] = pred
            fold_rmses.append(float(np.sqrt(mean_squared_error(y[va_idx], pred))))
            fold_maes.append(float(mean_absolute_error(y[va_idx], pred)))

        pooled_rmse = float(np.sqrt(mean_squared_error(y, oof)))
        result = {
            "target_transform": "raw",
            "candidate": cand_idx,
            "mean_fold_rmse": float(np.mean(fold_rmses)),
            "std_fold_rmse": float(np.std(fold_rmses, ddof=1)),
            "pooled_oof_rmse": pooled_rmse,
            "pooled_oof_mae": float(mean_absolute_error(y, oof)),
            "pooled_oof_r2": float(r2_score(y, oof)),
            **params,
        }
        all_results.append(result)
        print(f"[{run_no:02d}/{n_total}] target=raw   candidate={cand_idx:02d} "
              f"mean-CV-RMSE={result['mean_fold_rmse']:.4f} pooled-RMSE={pooled_rmse:.4f}", flush=True)
        # Checkpoint during search so partial progress is not lost if interrupted.
        pd.DataFrame(all_results).to_csv(BASE / "xgboost_cv_results.csv", index=False)
        if best_raw is None or result["mean_fold_rmse"] < best_raw["mean_fold_rmse"]:
            best_raw = {**result, "params": params.copy(), "oof": oof.copy(), "fold_rmses": fold_rmses, "fold_maes": fold_maes}

    # Compare raw target with log1p using the same winning tree parameters.
    params = best_raw["params"]
    run_no += 1
    oof_log = np.full(len(y), np.nan, dtype=float)
    fold_rmses_log, fold_maes_log = [], []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        model = make_pipeline(X.iloc[tr_idx], params)
        model.fit(X.iloc[tr_idx], np.log1p(y[tr_idx]))
        pred = np.expm1(np.clip(model.predict(X.iloc[va_idx]), -10, 15))
        pred = np.clip(pred, 0, None)
        oof_log[va_idx] = pred
        fold_rmses_log.append(float(np.sqrt(mean_squared_error(y[va_idx], pred))))
        fold_maes_log.append(float(mean_absolute_error(y[va_idx], pred)))
    log_result = {
        "target_transform": "log1p",
        "candidate": int(best_raw["candidate"]),
        "mean_fold_rmse": float(np.mean(fold_rmses_log)),
        "std_fold_rmse": float(np.std(fold_rmses_log, ddof=1)),
        "pooled_oof_rmse": float(np.sqrt(mean_squared_error(y, oof_log))),
        "pooled_oof_mae": float(mean_absolute_error(y, oof_log)),
        "pooled_oof_r2": float(r2_score(y, oof_log)),
        **params,
    }
    all_results.append(log_result)
    print(f"[{run_no:02d}/{n_total}] target=log1p candidate={int(best_raw['candidate']):02d} "
          f"mean-CV-RMSE={log_result['mean_fold_rmse']:.4f} pooled-RMSE={log_result['pooled_oof_rmse']:.4f}", flush=True)

    log_best = {**log_result, "params": params.copy(), "oof": oof_log.copy(), "fold_rmses": fold_rmses_log, "fold_maes": fold_maes_log}
    best = best_raw if best_raw["mean_fold_rmse"] <= log_best["mean_fold_rmse"] else log_best
    results_df = pd.DataFrame(all_results).sort_values("mean_fold_rmse").reset_index(drop=True)
    results_df.to_csv(BASE / "xgboost_cv_results.csv", index=False)
    pd.DataFrame([best_raw, log_best]).drop(columns=["params", "oof", "fold_rmses", "fold_maes"], errors="ignore").to_csv(
        BASE / "xgboost_transform_comparison.csv", index=False
    )

    # Refit winning configuration on all cleaned training rows.
    best_pipe = make_pipeline(X, best["params"])
    fit_y = np.log1p(y) if best["target_transform"] == "log1p" else y
    best_pipe.fit(X, fit_y)
    joblib.dump(best_pipe, BASE / "xgboost_flight_price_model.joblib", compress=3)
    try:
        transformed_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
        importances = best_pipe.named_steps["xgb"].feature_importances_
        importance_df = pd.DataFrame({"feature": transformed_names, "importance": importances})
        importance_df.sort_values("importance", ascending=False).to_csv(BASE / "xgboost_feature_importance.csv", index=False)
    except Exception as exc:
        print(f"Feature importance export skipped: {exc}", flush=True)
    test_pred = best_pipe.predict(X_test)
    if best["target_transform"] == "log1p":
        test_pred = np.expm1(np.clip(test_pred, -10, 15))
    test_pred = np.clip(test_pred, 0, None)

    # Follow the test id and sample-submission column order; sample values are placeholders.
    if sample is not None and "id" in sample.columns:
        out = sample[["id"]].merge(pd.DataFrame({"id": test["id"], "_pred": test_pred}), on="id", how="left", sort=False)
        if out["_pred"].isna().any():
            raise ValueError("Could not align every test id to its prediction.")
        submission = pd.DataFrame({"id": out["id"], "price": out["_pred"]})
    else:
        submission = pd.DataFrame({"id": test["id"], "price": test_pred})
    submission.to_csv(BASE / "submission_xgboost.csv", index=False)

    pd.DataFrame({"id": train["id"], "actual_price": y, "oof_prediction": best["oof"]}).to_csv(
        BASE / "xgboost_oof_predictions.csv", index=False
    )

    report = [
        "XGBoost flight-price regression validation report",
        "=================================================",
        f"Input training rows: {original_n}",
        f"Removed price <= 0 or missing: {removed_zero_or_invalid}",
        f"Removed price > {OUTLIER_PRICE_LIMIT:g}: {removed_extreme}",
        f"Rows used for training/CV: {len(train)}",
        f"Test rows predicted: {len(test)}",
        f"Validation: {N_SPLITS}-fold StratifiedKFold with {n_bins} quantile bins of price, shuffled, random_state={SEED}",
        f"Feature count before one-hot encoding: {X.shape[1]}",
        "Feature engineering: directional route, unordered city pair, route+stops interaction, ordinal stops, business flag, numeric flight number, log duration, log days-left.",
        "Categorical features are one-hot encoded within each fold; numeric features are median-imputed.",
        "",
        "Best configuration by mean fold RMSE (lower is better):",
        f"Target transform: {best['target_transform']}",
        f"Mean fold RMSE: {best['mean_fold_rmse']:.4f}",
        f"Std fold RMSE: {best['std_fold_rmse']:.4f}",
        f"Pooled out-of-fold RMSE: {best['pooled_oof_rmse']:.4f}",
        f"Pooled out-of-fold MAE: {best['pooled_oof_mae']:.4f}",
        f"Pooled out-of-fold R2: {best['pooled_oof_r2']:.5f}",
        f"Parameters: {best['params']}",
        "",
        "Top 5 search configurations:",
    ]
    for _, row in results_df.head(5).iterrows():
        report.append(f"{row['target_transform']} candidate {int(row['candidate'])}: mean CV RMSE={row['mean_fold_rmse']:.4f}, pooled OOF RMSE={row['pooled_oof_rmse']:.4f}")
    report += [
        "",
        "Note: these CV scores estimate generalization, but do not guarantee a Kaggle public leaderboard score.",
        f"Elapsed time: {time.time()-started:.1f} seconds",
    ]
    report_text = "\n".join(report)
    (BASE / "xgboost_validation_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text, flush=True)
    print("\nSaved: submission_xgboost.csv, xgboost_flight_price_model.joblib, xgboost_cv_results.csv, xgboost_oof_predictions.csv, xgboost_feature_importance.csv, xgboost_validation_report.txt", flush=True)


if __name__ == "__main__":
    main()
