import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from gustobot.application.foodtrace.cli import (
    render_demo_trace,
    run_foodtrace_case,
    run_foodtrace_incident,
)
from gustobot.application.foodtrace.fixtures import build_foodtrace_fixture


def test_demo_runs_undeclared_peanut_case_end_to_end():
    fixture = build_foodtrace_fixture()

    demo = run_foodtrace_case("allergen_peanut_001")

    assert demo.gate.state == "ALLOW_REPORT"
    assert demo.scope == fixture.gold_cases[0].gold_scope
    assert [step.name for step in demo.trace] == [
        "Incident Parser",
        "SOP Lookup",
        "Graph Scope",
        "Order Exposure",
        "Consistency Reviewer",
        "Decision Gate",
        "Recall Report",
    ]
    assert demo.report is not None
    assert demo.report.impacted_product_ids == (
        "PRODUCT-FT-001",
        "PRODUCT-FT-002",
    )
    assert demo.report.impacted_batch_ids == (
        "BATCH-FT-001",
        "BATCH-FT-002",
        "BATCH-FT-003",
    )
    assert demo.report.impacted_store_ids == (
        "STORE-FT-001",
        "STORE-FT-002",
        "STORE-FT-003",
    )
    assert demo.report.impacted_order_ids == (
        "ORDER-FT-001",
        "ORDER-FT-002",
        "ORDER-FT-003",
    )
    assert {item.evidence_id for item in demo.evidence}


def test_demo_trace_is_interview_readable_and_does_not_expose_local_secrets():
    rendered = render_demo_trace(run_foodtrace_case("allergen_peanut_001"))

    for stage in (
        "Incident Parser",
        "SOP Lookup",
        "Graph Scope",
        "Order Exposure",
        "Consistency Reviewer",
        "Decision Gate",
        "Recall Report",
    ):
        assert stage in rendered
    assert "Gate: ALLOW_REPORT" in rendered
    assert "BATCH-FT-001" in rendered
    assert "ORDER-FT-003" in rendered
    assert "EVD-" in rendered
    assert "password=" not in rendered.lower()
    assert "F:\\agent+" not in rendered


def test_demo_rejects_unknown_case_instead_of_silently_using_default():
    with pytest.raises(ValueError, match="unknown FoodTrace case"):
        run_foodtrace_case("does-not-exist")


def test_cli_sanitizes_unknown_case_without_traceback_or_local_path():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_foodtrace_demo.py",
            "--case",
            "does-not-exist",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "FoodTrace demo error: unknown case ID.\n"
    assert "Traceback" not in completed.stderr
    assert str(root) not in completed.stderr


def test_api_endpoint_is_a_thin_transform_over_the_same_workflow():
    from main import investigate_foodtrace

    incident = build_foodtrace_fixture().incident
    api_result = asyncio.run(investigate_foodtrace(incident))
    direct_result = run_foodtrace_incident(incident)

    assert api_result == direct_result
    assert api_result.gate.state == "ALLOW_REPORT"
