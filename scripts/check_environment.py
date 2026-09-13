"""Check local infrastructure and Model Studio configuration without exposing secrets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


class Results:
    def __init__(self) -> None:
        self.failed = 0

    def report(self, status: str, name: str, detail: str) -> None:
        print(f"[{status}] {name}: {detail}")
        if status == FAIL:
            self.failed += 1

    def run(self, name: str, check: Callable[[], str]) -> None:
        try:
            self.report(PASS, name, check())
        except Exception as exc:
            self.report(FAIL, name, _safe_error(exc))


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if api_key:
        text = text.replace(api_key, "<redacted>")
    return text[:300]


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is empty in .env")
    if "YOUR_WORKSPACE_ID" in value:
        raise RuntimeError(f"{name} still contains YOUR_WORKSPACE_ID")
    return value


def _post_json(url: str, api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def check_docker() -> str:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker is not installed")
    result = subprocess.run(
        [docker, "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError("Docker Desktop is installed but the engine is not running")
    return f"engine {result.stdout.strip()}"


def check_mysql() -> str:
    from sqlalchemy import create_engine, text

    engine = create_engine(_required("DATABASE_URL"), pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            value = connection.execute(text("SELECT 1")).scalar_one()
        return f"SELECT 1 -> {value}"
    finally:
        engine.dispose()


def check_neo4j() -> str:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        _required("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )
    try:
        driver.verify_connectivity()
        with driver.session(database="neo4j") as session:
            count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        return f"connected, nodes={count}"
    finally:
        driver.close()


def check_redis() -> str:
    import redis

    client = redis.from_url(_required("REDIS_URL"), decode_responses=True)
    try:
        client.ping()
        modules = client.module_list()
        names = sorted(
            str(item.get("name", item.get(b"name", "")))
            for item in modules
        )
        if not modules:
            raise RuntimeError("Redis is reachable but Redis Stack modules are missing")
        return f"ping ok, modules={','.join(names)}"
    finally:
        client.close()


def check_milvus() -> str:
    from pymilvus import connections, utility

    alias = "environment-check"
    connections.connect(
        alias=alias,
        host=_required("MILVUS_HOST"),
        port=_required("MILVUS_PORT"),
    )
    try:
        collections = utility.list_collections(using=alias)
        return f"connected, collections={collections or 'none yet'}"
    finally:
        connections.disconnect(alias)


def check_llm() -> str:
    key = _required("LLM_API_KEY")
    base = _required("LLM_BASE_URL").rstrip("/")
    model = _required("LLM_MODEL")
    data = _post_json(
        f"{base}/chat/completions",
        key,
        {
            "model": model,
            "messages": [{"role": "user", "content": "只回答 OK"}],
            "max_tokens": 8,
            "temperature": 0,
        },
    )
    if not data.get("choices"):
        raise RuntimeError(f"unexpected response: {str(data)[:200]}")
    return f"model={model}"


def check_embedding() -> str:
    key = _required("EMBEDDING_API_KEY")
    base = _required("EMBEDDING_BASE_URL").rstrip("/")
    model = _required("EMBEDDING_MODEL")
    data = _post_json(
        f"{base}/embeddings",
        key,
        {"model": model, "input": "宫保鸡丁"},
    )
    vector = data.get("data", [{}])[0].get("embedding", [])
    if not vector:
        raise RuntimeError(f"unexpected response: {str(data)[:200]}")
    return f"model={model}, dimensions={len(vector)}"


def check_rerank() -> str:
    key = _required("RERANK_API_KEY")
    base = _required("RERANK_BASE_URL").rstrip("/")
    endpoint = _required("RERANK_ENDPOINT")
    model = _required("RERANK_MODEL")
    data = _post_json(
        f"{base}/{endpoint.lstrip('/')}",
        key,
        {
            "model": model,
            "query": "宫保鸡丁的食材",
            "documents": ["鸡肉、花生和辣椒", "番茄和鸡蛋"],
            "top_n": 1,
        },
    )
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"unexpected response: {str(data)[:200]}")
    return f"model={model}, top_index={results[0].get('index')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--services", action="store_true", help="check Docker and databases")
    group.add_argument("--models", action="store_true", help="check Model Studio APIs")
    group.add_argument("--all", action="store_true", help="check everything (default)")
    args = parser.parse_args()

    check_services = args.services or args.all or not args.models
    check_models = args.models or args.all or not args.services
    results = Results()

    print(f"Environment: {ROOT}")
    print(f"Mode: {'demo' if os.getenv('DEEPREASON_DEMO_MODE', '').lower() == 'true' else 'real'}")
    if check_services:
        print("\nInfrastructure")
        for name, check in (
            ("Docker", check_docker),
            ("MySQL", check_mysql),
            ("Neo4j", check_neo4j),
            ("Redis Stack", check_redis),
            ("Milvus", check_milvus),
        ):
            results.run(name, check)
    if check_models:
        print("\nModel APIs")
        for name, check in (
            ("Qwen chat", check_llm),
            ("Embedding", check_embedding),
            ("Rerank", check_rerank),
        ):
            results.run(name, check)

    print(f"\nSummary: {results.failed} failed")
    return 1 if results.failed else 0


if __name__ == "__main__":
    sys.exit(main())
