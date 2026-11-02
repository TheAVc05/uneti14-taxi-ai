import argparse, json
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from preprocessing import FEATURE_COLUMNS, TARGET, make_features


def require_columns(frame, columns, label):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def validate_submission(submission, test):
    if list(submission.columns) != ["trip_id", TARGET]:
        raise ValueError("Submission must contain exactly trip_id, trip_duration in that order.")
    if len(submission) != len(test):
        raise ValueError(f"Submission has {len(submission)} rows; expected {len(test)}.")
    if not submission["trip_id"].equals(test["trip_id"]):
        raise ValueError("Submission trip_id values or order do not match the test set.")
    if submission["trip_id"].isna().any() or submission["trip_id"].duplicated().any():
        raise ValueError("Submission trip_id values must be present and unique.")
    durations = submission[TARGET].to_numpy(dtype=float)
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError("Submission predictions must be finite, positive durations.")

def rmsle(y_true, y_pred):
    y_pred = np.maximum(np.asarray(y_pred), 1e-6)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(np.asarray(y_true))) ** 2)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/UNETI2026_TRAIN_CHUNG_1200000.csv")
    parser.add_argument("--test", default="data/UNETI14_TEST_3000.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--n-estimators", type=int, default=3600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Evaluate a parameter set without fitting the final model or writing predictions.",
    )
    args = parser.parse_args()
    if args.n_estimators < 1 or args.learning_rate <= 0 or args.num_leaves < 2:
        parser.error("Model parameters must be positive (num-leaves must be at least 2).")
    if args.min_child_samples < 1:
        parser.error("--min-child-samples must be positive.")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train)
    require_columns(df, [*FEATURE_COLUMNS, "trip_id", TARGET], "Train")
    if df["trip_id"].isna().any() or df["trip_id"].duplicated().any():
        raise ValueError("Train trip_id values must be present and unique.")
    if df[TARGET].isna().any() or not np.isfinite(df[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Train target contains missing or non-finite values.")
    if (df[TARGET] <= 0).any():
        raise ValueError("Target has non-positive values; inspect before training.")

    X = make_features(df)
    y = df[TARGET].astype(float)
    tr_idx, va_idx = train_test_split(np.arange(len(df)), test_size=0.15, random_state=42)
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

    baseline = np.full(len(yva), float(ytr.median()))
    baseline_score = rmsle(yva, baseline)

    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples, subsample=0.85,
        subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    fit_options = {
        "eval_metric": "rmse",
        "callbacks": [lgb.early_stopping(60, verbose=True)],
    }
    if "eval_X" in inspect.signature(model.fit).parameters:
        model.fit(
            Xtr,
            np.log1p(ytr),
            eval_X=Xva,
            eval_y=np.log1p(yva),
            **fit_options,
        )
    else:
        model.fit(
            Xtr,
            np.log1p(ytr),
            eval_set=[(Xva, np.log1p(yva))],
            **fit_options,
        )
    val_pred = np.expm1(model.predict(Xva, num_iteration=model.best_iteration_))
    val_score = rmsle(yva, val_pred)
    report = {
        "train_rows": int(len(df)), "fit_rows": int(len(tr_idx)),
        "validation_rows": int(len(va_idx)), "feature_count": int(X.shape[1]),
        "baseline_median_rmsle": baseline_score, "lightgbm_rmsle": val_score,
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "random_state": 42,
        "parameters": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        },
    }
    (out / "validation_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.validation_only:
        return

    final_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(model.best_iteration_ or args.n_estimators),
        learning_rate=args.learning_rate, num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1
    )
    final_model.fit(X, np.log1p(y))
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path, compress=3)

    test = pd.read_csv(args.test)
    require_columns(test, [*FEATURE_COLUMNS, "trip_id"], "Test")
    if len(test) != 3000:
        raise ValueError(f"Test must contain exactly 3000 rows; received {len(test)}.")
    if test["trip_id"].isna().any() or test["trip_id"].duplicated().any():
        raise ValueError("Test trip_id values must be present and unique.")
    if df["trip_id"].isin(test["trip_id"]).any():
        raise ValueError("Train and test trip_id values must not overlap.")
    Xtest = make_features(test)
    pred = np.maximum(np.expm1(final_model.predict(Xtest)), 1.0)
    submission = pd.DataFrame({"trip_id": test["trip_id"], TARGET: pred})
    validate_submission(submission, test)
    submission.to_csv(out / "UNETI14_submission.csv", index=False)
    print("Saved model:", model_path)
    print("Saved submission:", out / "UNETI14_submission.csv")

if __name__ == "__main__":
    main()
