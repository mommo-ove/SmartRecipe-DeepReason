from gustobot.application.deepreason.evidence import EvidenceLedger
from gustobot.application.deepreason.models import EvidenceItem


def test_ledger_persists_unique_evidence(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    item = EvidenceItem.create(
        task_id="recipe-1", source_type="neo4j", source="graph", content="鸡肉菜谱"
    )

    ledger.append_many([item, item])

    assert [saved.evidence_id for saved in ledger.list()] == [item.evidence_id]

