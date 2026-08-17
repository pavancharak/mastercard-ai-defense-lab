"""Fast smoke tests against the real generate/data/transactions.csv (no
fixtures - Generate's own tests already validate that data; here we validate
that Defend's feature/split/train/eval pipeline runs correctly end to end on
it, with a small model so the test stays fast).

Scope: transactions.csv only. E1/disputes.csv is excluded this pass - see
defend/src/defend/data.py's module docstring for why (upstream ID-generation
bug in generate/entities.py blocks the join needed to recover its label).
"""

from defend.data import build_dataset
from defend.evaluate import evaluate
from defend.features import build_features
from defend.split import split_dataset


def test_dataset_has_four_full_categories_and_partial_e():
    df = build_dataset()
    categories = set(df.loc[df["is_fraud"].astype(bool), "fraud_category"].unique())
    assert categories == {"A", "B", "C", "D", "E"}
    # E1 is unrecoverable this pass; only E2 should appear under category E.
    e_vectors = set(df.loc[df["fraud_category"] == "E", "fraud_vector"].unique())
    assert e_vectors == {"E2"}


def test_split_preserves_rare_vectors():
    df = build_dataset()
    train_df, test_df = split_dataset(df, test_size=0.25, seed=1)
    for vector in ["C1", "C2", "C3"]:
        assert (train_df["fraud_vector"] == vector).sum() > 0
        assert (test_df["fraud_vector"] == vector).sum() > 0


def test_features_no_leakage_columns():
    df = build_dataset()
    X, y, spec = build_features(df)
    leaky = {"narrative_tag", "fraud_category", "fraud_vector", "is_fraud",
             "transaction_id", "account_id", "merchant_id"}
    assert leaky.isdisjoint(set(spec.feature_columns))


def test_end_to_end_train_and_evaluate_small():
    import xgboost as xgb

    df = build_dataset()
    train_df, test_df = split_dataset(df, test_size=0.3, seed=2)
    X_train, y_train, spec = build_features(train_df)
    X_test, y_test, _ = build_features(test_df, category_levels=spec.category_levels)

    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, enable_categorical=True, tree_method="hist")
    model.fit(X_train[spec.feature_columns], y_train)
    proba = model.predict_proba(X_test[spec.feature_columns])[:, 1]

    result = evaluate(test_df, proba)
    assert 0.0 <= result.aggregate["precision"] <= 1.0
    assert 0.0 <= result.aggregate["recall"] <= 1.0
    assert set(result.per_category["category"]) == {"A", "B", "C", "D", "E"}
    assert len(result.per_vector) == 11
    assert "E1" not in set(result.per_vector["vector"])
