"""Data models for custom security detection rules and rule matches."""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional
from pydantic import BaseModel, ConfigDict, Field


class RuleSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleMatch(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: str
    rule_name: str
    severity: str
    description: str
    mitre_technique: str = ""
    action: str = ""
    matched_value: str = ""


class RuleDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    severity: str
    description: str
    mitre_technique: str = ""
    action: str = ""
    enabled: bool = True
