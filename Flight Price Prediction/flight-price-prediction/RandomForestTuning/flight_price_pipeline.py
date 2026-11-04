"""Flight-price regression: leak-safe preprocessing pipeline + 5-fold CV.

Fits both a RandomForestRegressor and a CatBoostRegressor because these are
separate algorithms (CatBoost is gradient boosting, not a random forest).
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

RANDOM_STATE = 42
N_SPLITS = 5


def resolve_csv(filenames):
    script_dir = Path(__file__).resolve().parent

    for filename in filenames:
        candidate = script_dir / filename

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find any of these files in {script_dir}: {filenames}"
    )

TRAIN_PATH = resolve_csv(['train.csv', 'train(2).csv', 'train(1).csv'])
TEST_PATH = resolve_csv(['test.csv', 'test(2).csv', 'test(1).csv'])

class FlightFeatureEngineer(BaseEstimator, TransformerMixin):
    """Drop identifiers and engineer an origin-destination route feature."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=['id', 'flight', 'price'], errors='ignore')
        # Directional route interaction captures source/destination pairs.
        X['route'] = X['source_city'].astype(str) + ' -> ' + X['destination_city'].astype(str)
        # Small additional numeric feature for same-city trips.
        X['same_city'] = (X['source_city'].astype(str) == X['destination_city'].astype(str)).astype(int)
        return X


def flight_class_to_binary(X):
    """Binary encoding for the two classes: Economy=0, Business=1."""
    arr = np.asarray(X).reshape(-1)
    mapping = {'Economy': 0, 'Business': 1}
    encoded = np.array([mapping.get(str(v), -1) for v in arr], dtype=np.int8)
    return encoded.reshape(-1, 1)


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('airline_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['airline']),
            ('arrival_time_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['arrival_time']),
            # Categorical city and time features plus the engineered route.
            ('city_route_departure_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             ['source_city', 'destination_city', 'route', 'departure_time']),
            ('stops_ordinal', OrdinalEncoder(
                categories=[['zero', 'one', 'two_or_more']],
                handle_unknown='use_encoded_value', unknown_value=-1
            ), ['stops']),
            ('flight_class_binary', FunctionTransformer(flight_class_to_binary, validate=False), ['flight_class']),
            ('numeric', 'passthrough', ['duration', 'days_left', 'same_city']),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def make_pipeline(model):
    return Pipeline([
        ('feature_engineering', FlightFeatureEngineer()),
        ('preprocessing', make_preprocessor()),
        ('model', model),
    ])


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if 'price' not in train.columns:
        raise ValueError('train.csv must contain a price target column.')
    zero_price_count = int((train['price'] == 0).sum())
    train = train.loc[train['price'] != 0].copy()
    train = train.loc[train['price'].notna()].copy()

    X = train.drop(columns=['price'])
    y = train['price'].astype(float)
    X_test = test.copy()

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=350,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'CatBoost': CatBoostRegressor(
            iterations=600,
            depth=8,
            learning_rate=0.05,
            loss_function='RMSE',
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
        ),
    }

    rows = []
    fitted_pipelines = {}
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator)
        mae_folds = []
        rmse_folds = []
        r2_folds = []
        # Manual KFold loop is compatible with the installed CatBoost/sklearn
        # versions while still fitting preprocessing inside each fold.
        for fold_num, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            fold_pipeline = make_pipeline(estimator.__class__(**estimator.get_params()))
            X_fold_train, X_fold_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_fold_train, y_fold_valid = y.iloc[train_idx], y.iloc[valid_idx]
            fold_pipeline.fit(X_fold_train, y_fold_train)
            transformed_valid = fold_pipeline[:-1].transform(X_fold_valid)
            fold_pred = fold_pipeline.named_steps['model'].predict(transformed_valid)
            mae_folds.append(mean_absolute_error(y_fold_valid, fold_pred))
            rmse_folds.append(mean_squared_error(y_fold_valid, fold_pred) ** 0.5)
            r2_folds.append(r2_score(y_fold_valid, fold_pred))
        mae_folds = np.asarray(mae_folds)
        rmse_folds = np.asarray(rmse_folds)
        r2_folds = np.asarray(r2_folds)
        rows.append({
            'Model': name,
            'CV MAE mean': float(np.mean(mae_folds)),
            'CV MAE std': float(np.std(mae_folds, ddof=1)),
            'CV RMSE mean': float(np.mean(rmse_folds)),
            'CV R2 mean': float(np.mean(r2_folds)),
            'CV R2 std': float(np.std(r2_folds, ddof=1)),
            'Fold MAEs': ', '.join(f'{x:.2f}' for x in mae_folds),
        })
        # Refit on all eligible training rows and make predictions for test.csv.
        pipeline.fit(X, y)
        transformed_test = pipeline[:-1].transform(X_test)
        preds = np.maximum(0, pipeline.named_steps['model'].predict(transformed_test))
        fitted_pipelines[name] = pipeline
        out = pd.DataFrame({'id': test['id'] if 'id' in test.columns else np.arange(len(test)),
                            'price': np.round(preds, 2)})
        filename = 'submission_catboost.csv' if name == 'CatBoost' else 'submission_random_forest.csv'
        out.to_csv(BASE / filename, index=False)

    results = pd.DataFrame(rows).sort_values('CV MAE mean').reset_index(drop=True)
    results.to_csv(BASE / 'flight_model_cv_results.csv', index=False)
    best_name = str(results.iloc[0]['Model'])
    # Store the best-CV-MAE model's predictions under an easy-to-find filename.
    best_file = 'submission_catboost.csv' if best_name == 'CatBoost' else 'submission_random_forest.csv'
    pd.read_csv(BASE / best_file).to_csv(BASE / 'flight_price_predictions_best_model.csv', index=False)

    print(f'Training rows after removing price=0: {len(train):,}')
    print(f'Removed rows with price=0: {zero_price_count:,}')
    print(f'Test rows: {len(test):,} (price target not present; held-out test MAE/accuracy cannot be computed)')
    print(f'CV folds: {N_SPLITS}, shuffled, random_state={RANDOM_STATE}')
    print('\nCross-validation results (5-fold):')
    print(results.to_string(index=False, formatters={
        'CV MAE mean': '{:.2f}'.format,
        'CV MAE std': '{:.2f}'.format,
        'CV RMSE mean': '{:.2f}'.format,
        'CV R2 mean': '{:.4f}'.format,
        'CV R2 std': '{:.4f}'.format,
    }))
    print(f'\nBest by CV MAE: {best_name}')
    print(f'Predictions saved to: /mnt/data/flight_price_predictions_best_model.csv')
    print('Both model submissions saved to: /mnt/data/submission_catboost.csv and /mnt/data/submission_random_forest.csv')
    print('Metrics saved to: /mnt/data/flight_model_cv_results.csv')

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    main()
