from generate.config import DISPUTE_VECTORS, VECTOR_CATEGORY, VECTOR_WEIGHTS
from generate.dataset import build_dataset


def _small_result():
    return build_dataset(seed=7, scale=0.2, n_legit=200, n_dispute_legit=30)


def test_every_vector_present_and_labeled():
    result = _small_result()
    txn_vectors = set(result.transactions.loc[result.transactions.is_fraud, "fraud_vector"].unique())
    dispute_vectors = set(result.disputes.loc[result.disputes.is_fraud, "fraud_vector"].unique())
    seen = txn_vectors | dispute_vectors

    for vector in VECTOR_WEIGHTS:
        assert vector in seen, f"vector {vector} produced no rows"
        expected_table = "disputes" if vector in DISPUTE_VECTORS else "transactions"
        if expected_table == "transactions":
            assert vector in txn_vectors
        else:
            assert vector in dispute_vectors


def test_fraud_category_matches_vector():
    result = _small_result()
    fraud = result.transactions[result.transactions.is_fraud]
    for vector, category in fraud[["fraud_vector", "fraud_category"]].drop_duplicates().itertuples(index=False):
        assert VECTOR_CATEGORY[vector] == category


def test_legit_rows_unlabeled():
    result = _small_result()
    legit = result.transactions[~result.transactions.is_fraud]
    assert legit["fraud_category"].isna().all()
    assert legit["fraud_vector"].isna().all()
    assert len(legit) >= 200


def test_reproducible_with_seed():
    a = build_dataset(seed=123, scale=0.1, n_legit=50, n_dispute_legit=10)
    b = build_dataset(seed=123, scale=0.1, n_legit=50, n_dispute_legit=10)
    assert a.transactions.equals(b.transactions)
    assert a.disputes.equals(b.disputes)


def test_disputes_link_to_real_transactions():
    result = _small_result()
    txn_ids = set(result.transactions["transaction_id"])
    linked = set(result.disputes["original_transaction_id"])
    assert linked.issubset(txn_ids)


def test_card_testing_cluster_shape():
    result = _small_result()
    d1 = result.transactions[result.transactions.fraud_vector == "D1"]
    assert (d1["attempts_in_window"] > 1).all()
    # low authorization success rate, per taxonomy
    assert (d1["authorization_result"] == "approved").mean() < 0.3


def test_promo_abuse_shares_device_fingerprint_within_cluster():
    result = _small_result()
    e2 = result.transactions[result.transactions.fraud_vector == "E2"]
    devices_per_cluster = e2.groupby("account_cluster_id")["device_id"].nunique()
    assert (devices_per_cluster == 1).all()


def test_ids_unique_per_row():
    # regression test for the short_id() off-by-slice bug (fixed 2026-08-16):
    # every ID it produced collapsed to the same constant string, which made
    # transaction_id/dispute_id non-unique and silently turned the E1 dispute
    # join into a Cartesian product downstream in Defend.
    result = _small_result()
    assert result.transactions["transaction_id"].is_unique
    assert result.disputes["dispute_id"].is_unique


def test_credit_history_months_populated_for_every_vector():
    # regression test for the missingness-fingerprint bug (fixed 2026-08-16):
    # credit_history_months used to be null for every vector except A1/A2/
    # legit, which let a classifier learn "which generator wrote this row"
    # from presence/absence alone instead of genuine account-history signal.
    result = _small_result()
    key = result.transactions["fraud_vector"].fillna("LEGIT")
    non_null_rate = result.transactions.groupby(key)["credit_history_months"].apply(lambda s: s.notna().mean())
    assert (non_null_rate == 1.0).all(), non_null_rate[non_null_rate < 1.0]


def test_payee_fields_driven_by_channel_not_by_fraud_label():
    # regression test for the same missingness-fingerprint bug, applied to
    # payee_prior_interaction_count/is_first_time_payee: whether these are
    # populated must depend on the transaction's channel (does it even have
    # a payee?), not on whether the row is fraud or which vector wrote it.
    result = _small_result()
    payee_channels = {"wire", "ach", "rtp", "p2p"}
    has_payee_channel = result.transactions["channel"].isin(payee_channels)
    populated = result.transactions["payee_prior_interaction_count"].notna()
    assert (populated == has_payee_channel).all()
    # and both legit and fraud rows on payee channels actually get populated
    # (i.e. it's not one label-exclusive subset of those channels)
    on_payee_channel = result.transactions[has_payee_channel]
    assert on_payee_channel.loc[~on_payee_channel.is_fraud, "payee_prior_interaction_count"].notna().any()
    assert on_payee_channel.loc[on_payee_channel.is_fraud, "payee_prior_interaction_count"].notna().any()


def test_mandate_fields_populated_for_every_agentic_checkout_row():
    # regression test for the mandate-field missingness bug (fixed 2026-08-16):
    # C1/C2 set mandate_id from a real Agent but never propagated the rest of
    # its mandate (mandate_amount_cap/category_scope/recurring_flag, envelope
    # compliance), leaving transaction_within_mandate_envelope null for C1/C2
    # while legit/C3 always populated it - the same missingness fingerprint
    # as credit_history_months/payee_prior_interaction_count, just for the
    # agentic_checkout channel and only found once Category C was checked
    # directly after an unrelated fix incidentally regressed it to 1.000.
    result = _small_result()
    agentic = result.transactions[result.transactions["channel"] == "agentic_checkout"]
    for col in ["transaction_within_mandate_envelope", "agent_behavior_pattern_match_score"]:
        assert agentic[col].notna().all(), f"{col} has nulls on agentic_checkout rows"


def test_identity_verification_fields_populated_for_every_vector():
    # regression test for the identity-verification missingness bug (fixed
    # 2026-08-16, same session as its introduction): claimed_employment_tenure_months/
    # identity_cross_source_correlation were generalized to A2 first, in
    # isolation, and retraining immediately reproduced the same
    # null-vs-populated fingerprint as every earlier fix in this file - the
    # two fields carried ~69% combined Defend feature importance while every
    # vector but legit/A1/A2 read NaN. Caught before being reported and
    # fixed by applying the same helper to the remaining 9 generators.
    result = _small_result()
    key = result.transactions["fraud_vector"].fillna("LEGIT")
    for col in ["claimed_employment_tenure_months", "identity_cross_source_correlation"]:
        non_null_rate = result.transactions.groupby(key)[col].apply(lambda s: s.notna().mean())
        assert (non_null_rate == 1.0).all(), (col, non_null_rate[non_null_rate < 1.0])
