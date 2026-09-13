import pytest

from gustobot.application.foodtrace.sql_queries import build_exposure_queries


def test_exposure_queries_are_select_only_and_use_expanding_bound_parameters():
    queries = build_exposure_queries(["BATCH-FT-003", "BATCH-FT-001", "BATCH-FT-001"])

    assert queries.production_batch_ids == ("BATCH-FT-001", "BATCH-FT-003")
    for query in (queries.inventory, queries.orders):
        normalized = " ".join(query.statement.text.split()).upper()
        assert normalized.startswith("SELECT ")
        assert " PRODUCTION_BATCH_ID IN " in normalized
        assert query.parameters == {
            "production_batch_ids": ("BATCH-FT-001", "BATCH-FT-003")
        }
        assert query.statement._bindparams["production_batch_ids"].expanding is True
        assert all(
            keyword not in normalized
            for keyword in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ")
        )


@pytest.mark.parametrize(
    "invalid_scope",
    [[], (), "BATCH-FT-001", ["BATCH-FT-001", " "]],
)
def test_exposure_queries_reject_empty_or_malformed_batch_scope(invalid_scope):
    with pytest.raises(ValueError, match="production_batch_ids"):
        build_exposure_queries(invalid_scope)


def test_exposure_queries_reject_abnormally_large_scope():
    with pytest.raises(ValueError, match="100"):
        build_exposure_queries([f"BATCH-{index:03d}" for index in range(101)])
