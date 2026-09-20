"""Tests for custom security detection rules and rule engine."""
from __future__ import annotations

import pytest
from app.rules.engine import Rule, RuleEngine, RULE_ENGINE
from app.rules.models import RuleSeverity


def test_rule_definitions():
    defs = RULE_ENGINE.get_definitions()
    assert len(defs) >= 5
    ids = {d.id for d in defs}
    assert "RULE-SEC-001" in ids
    assert "RULE-SEC-002" in ids
    assert "RULE-SEC-003" in ids
    assert "RULE-AI-004" in ids
    assert "RULE-AI-005" in ids


def test_rule_toggle():
    eng = RuleEngine()
    assert eng.set_enabled("RULE-SEC-001", False) is True
    defs = {d.id: d.enabled for d in eng.get_definitions()}
    assert defs["RULE-SEC-001"] is False
    assert eng.set_enabled("NON_EXISTENT_RULE", True) is False


def test_rule_evaluation_syn_scan():
    eng = RuleEngine()
    # Benign window
    matches = eng.evaluate_anchor(
        anchor={"alert": False, "max_attack_prob": 0.05, "horizon": []},
        window_features={"syn_ratio": 0.05, "flow_count": 50},
    )
    assert not any(m.rule_id == "RULE-SEC-001" for m in matches)

    # Attack window: high syn ratio + volume
    matches = eng.evaluate_anchor(
        anchor={"alert": True, "max_attack_prob": 0.85, "horizon": []},
        window_features={"syn_ratio": 0.92, "flow_count": 150},
    )
    rule_ids = {m.rule_id for m in matches}
    assert "RULE-SEC-001" in rule_ids
    match = next(m for m in matches if m.rule_id == "RULE-SEC-001")
    assert match.severity == "HIGH"
    assert "92.0%" in match.matched_value


def test_rule_evaluation_ai_onset():
    eng = RuleEngine()
    anchor = {
        "alert": True,
        "max_attack_prob": 0.82,
        "lead_time_seconds": 30,
        "horizon": [
            {
                "k": 1,
                "horizon_seconds": 10,
                "progression_state": "ONSET",
                "attack_prob": 0.78,
            }
        ],
    }
    matches = eng.evaluate_anchor(anchor=anchor, window_features={})
    rule_ids = {m.rule_id for m in matches}
    assert "RULE-AI-004" in rule_ids


@pytest.mark.asyncio
async def test_rules_api_direct():
    from app.api.routes_rules import list_rules, toggle_rule, RuleToggleRequest
    rules = await list_rules()
    assert len(rules) >= 5

    # Toggle
    res = await toggle_rule(RuleToggleRequest(rule_id="RULE-SEC-001", enabled=False))
    rule_map = {x.id: x.enabled for x in res}
    assert rule_map["RULE-SEC-001"] is False

    # Toggle back
    res = await toggle_rule(RuleToggleRequest(rule_id="RULE-SEC-001", enabled=True))
    rule_map = {x.id: x.enabled for x in res}
    assert rule_map["RULE-SEC-001"] is True

