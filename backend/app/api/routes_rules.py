"""API endpoints for custom and built-in detection rules."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rules.engine import RULE_ENGINE
from app.rules.models import RuleDefinition

router = APIRouter(prefix="/rules", tags=["rules"])


class RuleToggleRequest(BaseModel):
    rule_id: str
    enabled: bool


@router.get("", response_model=list[RuleDefinition])
async def list_rules() -> list[RuleDefinition]:
    """List all configured detection rules and their current operational status."""
    return RULE_ENGINE.get_definitions()


@router.post("/toggle", response_model=list[RuleDefinition])
async def toggle_rule(body: RuleToggleRequest) -> list[RuleDefinition]:
    """Enable or disable a specific detection rule."""
    success = RULE_ENGINE.set_enabled(body.rule_id, body.enabled)
    if not success:
        raise HTTPException(status_code=404, detail=f"Rule '{body.rule_id}' not found")
    return RULE_ENGINE.get_definitions()
