from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from gustobot.application.deepreason.models import WorkflowResult
from gustobot.application.services.deepreason_service import get_orchestrator
from gustobot.domain.models.schemas import ChatRequest
from gustobot.infrastructure.core.logger import get_logger


logger = get_logger(service="http.deepreason")
router = APIRouter(prefix="/deepreason", tags=["DeepReason 多智能体"])


@router.post("/chat", response_model=WorkflowResult, summary="DeepReason 多智能体对话入口")
async def deepreason_chat_query(req: ChatRequest) -> WorkflowResult:
    try:
        return await get_orchestrator().run(
            req.message,
            session_id=req.session_id,
            user_id=req.user_id,
            image_path=req.image_path,
            file_path=req.file_path,
        )
    except RuntimeError as exc:
        logger.warning("DeepReason 未就绪: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("DeepReason 对话失败: %s", exc)
        raise HTTPException(status_code=500, detail="多智能体工作流执行失败") from exc


@router.get("/status", summary="DeepReason 运行状态")
async def deepreason_status() -> dict[str, Any]:
    orchestrator = get_orchestrator()
    return {
        "initialized": True,
        "domains": [domain.value for domain in orchestrator.registry.domains],
        "last_run": orchestrator.last_result.model_dump(mode="json") if orchestrator.last_result else None,
    }

