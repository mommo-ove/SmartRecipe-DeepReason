from pathlib import Path

from smartrecipe_eval_ui.app import build_app


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "benchmark" / "meal_planning"


if __name__ == "__main__":
    build_app(project_root=ROOT).queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        allowed_paths=[str(REPORT_ROOT)],
    )
