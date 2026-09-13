import inspect
from pathlib import Path

import gradio as gr

from smartrecipe_eval_ui.app import (
    build_app,
    create_live_run_callback,
    default_case_id,
    default_model_path,
)


def test_gradio_app_exposes_real_run_case_inspection_and_interview_tabs(tmp_path: Path):
    app = build_app(project_root=tmp_path)

    assert isinstance(app, gr.Blocks)
    config = app.get_config_file()
    rendered = str(config)
    assert "运行真实评测" in rendered
    assert "逐题检查" in rendered
    assert "Text2Cypher专项" in rendered
    assert "100问" in rendered


def test_default_model_path_resolves_from_worktree_back_to_workspace_cache():
    root = Path("F:/workspace/SmartRecipe/.worktrees/feature")

    model = default_model_path(root)

    assert model == Path(
        "F:/workspace/.hf-cache/hub/models--BAAI--bge-m3/"
        "snapshots/5617a9f61b028005a4858fdac845db406aefb181"
    )


def test_default_case_prefers_the_first_frozen_test_case():
    cases = [
        {"case_id": "dev-1", "split": "development"},
        {"case_id": "test-2", "split": "test"},
        {"case_id": "test-1", "split": "test"},
    ]

    assert default_case_id(cases) == "test-1"


def test_live_run_callback_is_a_generator_function(tmp_path: Path):
    callback = create_live_run_callback(tmp_path)

    assert inspect.isgeneratorfunction(callback)
