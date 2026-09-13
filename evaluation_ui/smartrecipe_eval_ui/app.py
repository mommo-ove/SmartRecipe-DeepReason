from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import socket
import subprocess
from typing import Any, Iterator
from urllib.parse import urlparse

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from .core import (
    build_backend_command,
    build_text2cypher_backend_command,
    inspect_case,
    inspect_text2cypher_case,
    load_interview_questions,
    load_saved_reports,
    load_text2cypher_report,
    metric_rows,
    text2cypher_metric_rows,
    text2cypher_report_case_ids,
    text2cypher_report_configurations,
)


CONFIGURATIONS = [
    "bm25",
    "dense_bge_m3",
    "rrf_bm25_dense",
    "rrf_with_graph_gate",
]

TEXT2CYPHER_CONFIGURATIONS = [
    "deepseek_direct",
    "deepseek_schema",
    "deepseek_schema_fewshot",
    "template_first_validated",
]


def default_model_path(project_root: Path) -> Path:
    worktrees_dir = next(
        (parent for parent in project_root.parents if parent.name == ".worktrees"),
        None,
    )
    workspace = (
        worktrees_dir.parent.parent
        if worktrees_dir is not None
        else project_root.parent
    )
    return (
        workspace
        / ".hf-cache"
        / "hub"
        / "models--BAAI--bge-m3"
        / "snapshots"
        / "5617a9f61b028005a4858fdac845db406aefb181"
    )


def default_case_id(cases: list[dict[str, Any]]) -> str | None:
    test_ids = sorted(
        str(item["case_id"]) for item in cases if item.get("split") == "test"
    )
    if test_ids:
        return test_ids[0]
    all_ids = sorted(str(item["case_id"]) for item in cases)
    return all_ids[0] if all_ids else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def _result_view(run_dir: Path):
    snapshot = load_saved_reports(run_dir)
    rows = metric_rows(snapshot)
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = frame["configuration"]
    axes[0].bar(labels, frame["recall@20"], color="#2563eb")
    axes[0].set_title("Recall@20（越高越好）")
    axes[0].set_ylim(0, 105)
    axes[1].bar(labels, frame["violation@20"], color="#dc2626")
    axes[1].set_title("禁用食材违规率@20（越低越好）")
    axes[1].set_ylim(0, max(35, float(frame["violation@20"].max()) + 5))
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    status = (
        f"已加载 **{len(rows)}** 组真实结果；split=`{snapshot.split}`，"
        f"语料={snapshot.corpus_count}条，测试集={snapshot.benchmark_count}条。"
    )
    return status, frame, figure


def _check_environment(
    project_root: Path, model_path: str, neo4j_url: str
) -> dict[str, Any]:
    model = Path(model_path)
    model_ready = model.is_dir() and (model / "pytorch_model.bin").exists()
    parsed = urlparse(neo4j_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    neo4j_ready = False
    error = ""
    try:
        with socket.create_connection((host, port), timeout=2):
            neo4j_ready = True
    except OSError as exc:
        error = str(exc)
    report_dir = (
        project_root / "benchmark" / "meal_planning" / "graph_relations" / "runs"
    )
    return {
        "backend_project": project_root.exists(),
        "bge_model_ready": model_ready,
        "bge_model_path": str(model),
        "neo4j_tcp_ready": neo4j_ready,
        "neo4j_url": neo4j_url,
        "neo4j_error": error,
        "saved_report_count": len(list(report_dir.glob("*.json"))),
    }


def _build_graph(project_root: Path, neo4j_url: str) -> str:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/build_meal_planning_graph.py",
            "--neo4j-url",
            neo4j_url,
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode:
        return f"建图失败（exit={completed.returncode}）：\n\n```text\n{output[-3000:]}\n```"
    return f"真实Neo4j建图完成：\n\n```json\n{output}\n```"


def _run_live(
    project_root: Path,
    configurations: list[str],
    split: str,
    model_path: str,
    neo4j_url: str,
) -> Iterator[tuple[str, pd.DataFrame, Any, str | None]]:
    empty = pd.DataFrame()
    yield "正在启动后端真实评测，请保留页面。BGE-M3在CPU上约需1～3分钟。", empty, None, None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        project_root
        / "benchmark"
        / "meal_planning"
        / "graph_relations"
        / "runs"
        / "ui"
        / split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evaluation-{stamp}.json"
    try:
        command = build_backend_command(
            project_root=project_root,
            configurations=list(configurations or []),
            model_path=Path(model_path),
            neo4j_url=neo4j_url,
            split=split,
            output_path=output_path,
        )
        completed = subprocess.run(
            command,
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=1200,
            check=False,
        )
    except Exception as exc:
        yield f"评测启动失败：`{exc}`", empty, None, None
        return
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-4000:]
        yield (
            f"评测失败（exit={completed.returncode}）：\n\n```text\n{detail}\n```",
            empty,
            None,
            None,
        )
        return
    status, frame, figure = _result_view(output_dir)
    yield f"真实评测完成。{status}", frame, figure, str(output_path)


def create_live_run_callback(project_root: Path):
    """Return a real generator callback so Gradio can stream progress updates."""

    def run_live_callback(configs, split, model, neo4j):
        yield from _run_live(project_root, configs, split, model, neo4j)

    return run_live_callback


def _latest_text2cypher_report(run_dir: Path) -> Path:
    reports = sorted(run_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        raise ValueError(f"no Text2Cypher reports found in {run_dir}")
    return reports[-1]


def _text2cypher_result_view(report_path: Path):
    report = load_text2cypher_report(report_path)
    frame = pd.DataFrame(text2cypher_metric_rows(report))
    case_ids = text2cypher_report_case_ids(report)
    configurations = text2cypher_report_configurations(report)
    status = (
        f"已加载 `{report_path.name}`；split=`{report['split']}`，"
        f"model=`{report.get('model')}`，prompt=`{report.get('prompt_version')}`。"
    )
    return (
        status,
        frame,
        str(report_path),
        report.get("security", {}),
        gr.update(choices=case_ids, value=case_ids[0] if case_ids else None),
        gr.update(
            choices=configurations,
            value=configurations[0] if configurations else None,
        ),
    )


def _run_text2cypher(
    project_root: Path,
    configurations: list[str],
    split: str,
    neo4j_url: str,
):
    output_dir = (
        project_root / "benchmark" / "meal_planning" / "text2cypher" / "runs" / "ui"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"{split}-{stamp}.json"
    command = build_text2cypher_backend_command(
        project_root=project_root,
        configurations=list(configurations or []),
        neo4j_url=neo4j_url,
        split=split,
        output_path=output_path,
    )
    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-4000:]
        return (
            f"Text2Cypher评测失败：\n\n```text\n{detail}\n```",
            pd.DataFrame(),
            None,
            {},
            gr.update(),
            gr.update(),
        )
    return _text2cypher_result_view(output_path)


def build_app(*, project_root: Path) -> gr.Blocks:
    project_root = project_root.resolve()
    run_dir = project_root / "benchmark" / "meal_planning" / "graph_relations" / "runs"
    cases_path = (
        project_root
        / "benchmark"
        / "meal_planning"
        / "graph_relations"
        / "retrieval_cases.v1.jsonl"
    )
    corpus_path = (
        project_root
        / "gustobot"
        / "data"
        / "meal_planning"
        / "external"
        / "retrieval_corpus.v1.jsonl"
    )
    question_path = project_root / "docs" / "interview" / "smartrecipe-100-questions.md"
    cases = _read_jsonl(cases_path)
    case_by_id = {item["case_id"]: item for item in cases}
    documents = {item["recipe_id"]: item for item in _read_jsonl(corpus_path)}
    questions = (
        load_interview_questions(question_path) if question_path.exists() else []
    )
    default_model = default_model_path(project_root)
    run_live_callback = create_live_run_callback(project_root)
    text2cypher_run_dir = (
        project_root / "benchmark" / "meal_planning" / "text2cypher" / "runs"
    )
    text2cypher_cases = _read_jsonl(
        project_root
        / "benchmark"
        / "meal_planning"
        / "text2cypher"
        / "cases.v1.jsonl"
    )

    css = """
    .hero {padding: 18px 22px; border-radius: 16px; background: linear-gradient(120deg,#eff6ff,#f5f3ff);}
    .note {border-left: 4px solid #2563eb; padding-left: 12px;}
    """
    with gr.Blocks(title="SmartRecipe Evaluation Console", css=css) as demo:
        gr.Markdown(
            "# SmartRecipe 真实评测控制台\n"
            "这里的按钮会调用后端脚本重新编码、检索和查询Neo4j；页面不写死评测数字。",
            elem_classes="hero",
        )
        with gr.Tab("运行真实评测"):
            with gr.Row():
                model_input = gr.Textbox(
                    value=str(default_model), label="BGE-M3本地模型目录", scale=3
                )
                neo4j_input = gr.Textbox(
                    value="bolt://localhost:17687", label="Neo4j Bolt地址", scale=2
                )
            with gr.Row():
                config_input = gr.CheckboxGroup(
                    choices=CONFIGURATIONS,
                    value=CONFIGURATIONS,
                    label="消融配置",
                )
                split_input = gr.Radio(
                    choices=["development", "test"],
                    value="test",
                    label="评测数据划分",
                )
            with gr.Row():
                check_button = gr.Button("检查环境")
                graph_button = gr.Button("导入/刷新Neo4j图谱")
                run_button = gr.Button("开始真实评测", variant="primary")
            environment_output = gr.JSON(label="环境状态")
            graph_output = gr.Markdown(elem_classes="note")
            run_status = gr.Markdown("尚未运行。", elem_classes="note")
            metrics_output = gr.Dataframe(label="真实指标", interactive=False)
            plot_output = gr.Plot(label="指标对照")
            report_output = gr.File(label="原始JSON报告")
            check_button.click(
                fn=lambda model, neo4j: _check_environment(project_root, model, neo4j),
                inputs=[model_input, neo4j_input],
                outputs=environment_output,
            )
            graph_button.click(
                fn=lambda neo4j: _build_graph(project_root, neo4j),
                inputs=neo4j_input,
                outputs=graph_output,
            )
            run_button.click(
                fn=run_live_callback,
                inputs=[config_input, split_input, model_input, neo4j_input],
                outputs=[run_status, metrics_output, plot_output, report_output],
                concurrency_limit=1,
            )

        with gr.Tab("已保存结果"):
            refresh_button = gr.Button("读取已有真实报告")
            saved_status = gr.Markdown()
            saved_metrics = gr.Dataframe(interactive=False)
            saved_plot = gr.Plot()

            def refresh_saved():
                try:
                    return _result_view(run_dir)
                except Exception as exc:
                    return f"读取失败：`{exc}`", pd.DataFrame(), None

            refresh_button.click(
                fn=refresh_saved,
                outputs=[saved_status, saved_metrics, saved_plot],
            )

        with gr.Tab("逐题检查"):
            gr.Markdown("选择同一道题和不同检索配置，观察Gold命中与禁用食材违规。")
            with gr.Row():
                case_input = gr.Dropdown(
                    choices=sorted(case_by_id),
                    value=default_case_id(cases),
                    label="测试题",
                )
                case_config = gr.Dropdown(
                    choices=CONFIGURATIONS,
                    value="bm25",
                    label="检索配置",
                )
                inspect_button = gr.Button("检查这一题", variant="primary")
            case_summary = gr.JSON(label="问题与约束")
            case_ranking = gr.Dataframe(label="Top-20结果", interactive=False)

            def inspect_selected(case_id: str, configuration: str):
                try:
                    snapshot = load_saved_reports(run_dir)
                    detail = inspect_case(
                        snapshot, configuration, case_by_id[case_id], documents
                    )
                    return (
                        {
                            key: value
                            for key, value in detail.items()
                            if key != "ranking_rows"
                        },
                        pd.DataFrame(detail["ranking_rows"]),
                    )
                except Exception as exc:
                    return {"error": str(exc)}, pd.DataFrame()

            inspect_button.click(
                fn=inspect_selected,
                inputs=[case_input, case_config],
                outputs=[case_summary, case_ranking],
            )

        with gr.Tab("Text2Cypher专项"):
            gr.Markdown(
                "运行四组生成消融，检查逻辑准确率、Schema/EXPLAIN通过率、"
                "修复Trace和独立安全攻击集；指标均从后端JSON报告读取。"
            )
            with gr.Row():
                t2c_configs = gr.CheckboxGroup(
                    choices=TEXT2CYPHER_CONFIGURATIONS,
                    value=["template_first_validated"],
                    label="配置（DeepSeek配置会真实调用API）",
                )
                t2c_split = gr.Radio(
                    choices=["development", "test"],
                    value="development",
                    label="数据划分",
                )
                t2c_neo4j = gr.Textbox(
                    value="bolt://localhost:17687", label="Neo4j Bolt地址"
                )
            with gr.Row():
                t2c_run = gr.Button("运行Text2Cypher评测", variant="primary")
                t2c_refresh = gr.Button("读取最近报告")
            t2c_status = gr.Markdown("尚未运行。", elem_classes="note")
            t2c_metrics = gr.Dataframe(label="专项指标", interactive=False)
            t2c_report_file = gr.File(label="原始JSON报告")
            t2c_security = gr.JSON(label="安全攻击集结果")
            with gr.Row():
                t2c_case = gr.Dropdown(
                    choices=sorted(item["case_id"] for item in text2cypher_cases),
                    value=default_case_id(text2cypher_cases),
                    label="Case",
                )
                t2c_case_config = gr.Dropdown(
                    choices=TEXT2CYPHER_CONFIGURATIONS,
                    value="template_first_validated",
                    label="配置",
                )
                t2c_inspect = gr.Button("查看Trace")
            t2c_detail = gr.JSON(label="候选Cypher、验证报告与Trace")
            t2c_run.click(
                fn=lambda configs, split, neo4j: _run_text2cypher(
                    project_root, configs, split, neo4j
                ),
                inputs=[t2c_configs, t2c_split, t2c_neo4j],
                outputs=[
                    t2c_status,
                    t2c_metrics,
                    t2c_report_file,
                    t2c_security,
                    t2c_case,
                    t2c_case_config,
                ],
                concurrency_limit=1,
            )

            def refresh_text2cypher():
                try:
                    return _text2cypher_result_view(
                        _latest_text2cypher_report(text2cypher_run_dir)
                    )
                except Exception as exc:
                    return (
                        f"读取失败：`{exc}`",
                        pd.DataFrame(),
                        None,
                        {},
                        gr.update(),
                        gr.update(),
                    )

            t2c_refresh.click(
                fn=refresh_text2cypher,
                outputs=[
                    t2c_status,
                    t2c_metrics,
                    t2c_report_file,
                    t2c_security,
                    t2c_case,
                    t2c_case_config,
                ],
            )
            def inspect_text2cypher(case_id: str, configuration: str):
                try:
                    report = load_text2cypher_report(
                        _latest_text2cypher_report(text2cypher_run_dir)
                    )
                    return inspect_text2cypher_case(report, configuration, case_id)
                except Exception as exc:
                    return {"error": str(exc)}

            t2c_inspect.click(
                fn=inspect_text2cypher,
                inputs=[t2c_case, t2c_case_config],
                outputs=t2c_detail,
            )

        with gr.Tab("项目100问"):
            with gr.Row():
                question_search = gr.Textbox(
                    label="搜索问题或答案", placeholder="例如：RRF、CP-SAT、depends_on"
                )
                category_input = gr.Dropdown(
                    choices=["全部", *sorted({item["category"] for item in questions})],
                    value="全部",
                    label="分类",
                )
                search_button = gr.Button("搜索", variant="primary")
            question_output = gr.Markdown(
                f"题库已加载 **{len(questions)}** 题。输入关键词开始复习。"
            )

            def search_bank(keyword: str, category: str) -> str:
                token = (keyword or "").strip().casefold()
                matches = [
                    item
                    for item in questions
                    if (category == "全部" or item["category"] == category)
                    and (
                        not token
                        or token in item["question"].casefold()
                        or token in item["answer"].casefold()
                    )
                ][:20]
                if not matches:
                    return "没有匹配的问题。"
                return "\n\n---\n\n".join(
                    f"### {item['number']:03d}. {item['question']}\n\n{item['answer']}"
                    for item in matches
                )

            search_button.click(
                fn=search_bank,
                inputs=[question_search, category_input],
                outputs=question_output,
            )
            question_search.submit(
                fn=search_bank,
                inputs=[question_search, category_input],
                outputs=question_output,
            )
    return demo
