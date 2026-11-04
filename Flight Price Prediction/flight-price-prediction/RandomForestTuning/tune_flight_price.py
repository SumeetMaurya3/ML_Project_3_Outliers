"""Tune Random Forest and CatBoost for flight-price regression with 5-fold CV.

This script reuses the preprocessing pipeline defined in flight_price_pipeline.py.
It selects hyperparameters by minimizing cross-validation MAE.

Put this file beside flight_price_pipeline.py, train.csv, and test.csv, then run:
    python tune_flight_price.py
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from catboost import CatBoostRegressor

from flight_price_pipeline import make_pipeline, RANDOM_STATE, N_SPLITS

BASE = Path(__file__).resolve().parent
N_JOBS = 4  # safer for a laptop; increase if your machine has spare RAM/CPU


def resolve_csv(names):
    for name in names:
        candidate = BASE / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find any of {names} in {BASE}")


def main():
    train_path = resolve_csv(["train.csv", "train(2).csv", "train(1).csv"])
    test_path = resolve_csv(["test.csv", "test(2).csv", "test(1).csv"])

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if "price" not in train.columns:
        raise ValueError("Training data must contain a 'price' column.")

    n_zero = int((train["price"] == 0).sum())
    train = train.loc[train["price"].notna() & (train["price"] != 0)].copy()
    X = train.drop(columns=["price"])
    y = train["price"].astype(float)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Random Forest search space
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    rf_params = {
        "model__n_estimators": [250, 350, 450, 600, 800],
        "model__max_depth": [None, 12, 16, 20, 24, 30],
        "model__min_samples_split": [2, 5, 10, 15],
        "model__min_samples_leaf": [1, 2, 3, 5, 8],
        "model__max_features": [0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        "model__bootstrap": [True, False],
    }

    # CatBoost search space. CatBoost is gradient boosting, not Random Forest.
    cat = CatBoostRegressor(
        loss_function="MAE",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
    )
    cat_params = {
        "model__iterations": [400, 600, 800, 1000],
        "model__depth": [4, 5, 6, 7, 8, 9, 10],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "model__l2_leaf_reg": [3, 5, 7, 9],
        "model__random_strength": [0.5, 1, 2, 5],
    }

    searches = {
        "Random Forest": RandomizedSearchCV(
            estimator=make_pipeline(rf),
            param_distributions=rf_params,
            n_iter=24,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=2,
            refit=True,
            return_train_score=True,
            pre_dispatch=N_JOBS,
        ),
        "CatBoost": RandomizedSearchCV(
            estimator=make_pipeline(cat),
            param_distributions=cat_params,
            n_iter=18,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=2,
            refit=True,
            return_train_score=True,
            pre_dispatch=N_JOBS,
        ),
    }

    summary = []
    fitted_searches = {}
    for model_name, search in searches.items():
        print(f"\n===== Tuning {model_name} =====")
        search.fit(X, y)
        fitted_searches[model_name] = search
        best_cv_mae = -float(search.best_score_)
        best_index = int(search.best_index_)
        std_neg_mae = float(search.cv_results_["std_test_score"][best_index])
        train_mae = -float(search.cv_results_["mean_train_score"][best_index])

        print(f"Best 5-fold CV MAE: {best_cv_mae:.4f} (fold-score std: {std_neg_mae:.4f})")
        print(f"Mean training MAE for best setting: {train_mae:.4f}")
        print("Best parameters:")
        print(json.dumps(search.best_params_, indent=2, default=str))

        summary.append({
            "Model": model_name,
            "Best CV MAE": best_cv_mae,
            "CV MAE score std": std_neg_mae,
            "Mean training MAE": train_mae,
            "Best parameters": json.dumps(search.best_params_, default=str),
        })

        # RandomizedSearchCV(refit=True) refits the best pipeline on all eligible rows.
        best_pipeline = search.best_estimator_
        test_predictions = np.maximum(0, best_pipeline.predict(test))
        ids = test["id"] if "id" in test.columns else np.arange(len(test))
        output_name = "tuned_random_forest_predictions.csv" if model_name == "Random Forest" else "tuned_catboost_predictions.csv"
        pd.DataFrame({"id": ids, "price": np.round(test_predictions, 2)}).to_csv(BASE / output_name, index=False)
        print(f"Saved predictions: {BASE / output_name}")

    summary_df = pd.DataFrame(summary).sort_values("Best CV MAE").reset_index(drop=True)
    summary_df.to_csv(BASE / "tuned_model_cv_summary.csv", index=False)
    best_name = str(summary_df.iloc[0]["Model"])
    best_file = "tuned_random_forest_predictions.csv" if best_name == "Random Forest" else "tuned_catboost_predictions.csv"
    pd.read_csv(BASE / best_file).to_csv(BASE / "tuned_best_model_predictions.csv", index=False)

    print("\n===== Summary =====")
    print(summary_df[["Model", "Best CV MAE", "CV MAE score std", "Mean training MAE"]].to_string(index=False))
    print(f"\nBest model by 5-fold CV MAE: {best_name}")
    print(f"Removed rows with price=0: {n_zero}")
    print(f"Eligible training rows: {len(train)}; unseen test rows: {len(test)}")
    print("Note: test.csv has no true price labels, so test MAE/R² cannot be calculated.")
    print(f"Summary saved to: {BASE / 'tuned_model_cv_summary.csv'}")
    print(f"Best-model predictions saved to: {BASE / 'tuned_best_model_predictions.csv'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
