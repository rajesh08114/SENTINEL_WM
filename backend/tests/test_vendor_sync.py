"""Guard against drift between vendored app/sentinel_infer/ and the research
source. Runs only in the monorepo (../research present).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_RESEARCH = Path(__file__).resolve().parents[2] / "research" / "sentinel_wm"
_VENDOR = Path(__file__).resolve().parents[1] / "app" / "sentinel_infer"
pytestmark = pytest.mark.skipif(
    not _RESEARCH.is_dir(), reason="research/ not checked out")


def _assigned_lists(pyfile: Path, names) -> dict:
    """{name: literal} for top-level `name = [...]` and `name: T = [...]`."""
    tree = ast.parse(pyfile.read_text())
    out = {}
    for node in tree.body:
        tgt = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            tgt = node.target
        if isinstance(tgt, ast.Name) and tgt.id in names and node.value is not None:
            try:
                out[tgt.id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


def _attack_table(pyfile: Path) -> dict:
    """{family: (tactic, techniques, kill_chain_phase, base_confidence)} parsed
    from the `CIC_LABEL_TO_ATTACK = {...}` literal via a stub AttackRef."""
    tree = ast.parse(pyfile.read_text())
    for node in tree.body:
        tgt = (node.targets[0] if isinstance(node, ast.Assign) and node.targets
               else getattr(node, "target", None))
        if isinstance(tgt, ast.Name) and tgt.id == "CIC_LABEL_TO_ATTACK":
            def AttackRef(t, x, p, c, note=""):
                return (t, tuple(x), p, c)
            _NONE = ("None", (), "None", "High")
            return eval(compile(ast.Expression(node.value), "<t>", "eval"),
                        {"AttackRef": AttackRef, "_NONE": _NONE})
    raise KeyError("CIC_LABEL_TO_ATTACK")


def test_locked_feature_schema_matches_research():
    from app.sentinel_infer import schema as vend
    keys = ["IDENTITY_COLS", "TIER1_FLOW_COLS", "TIER2_PACKET_COLS",
            "TRUE_FLAG_COLS", "PROGRESSION_STATES"]
    src = _assigned_lists(_RESEARCH / "config.py", keys)
    for k in keys:
        assert src.get(k) == getattr(vend, k), f"{k} drifted from research/config.py"


def test_state_feature_cols_match_research():
    from app.sentinel_infer.windows import STATE_FEATURE_COLS
    src = _assigned_lists(
        _RESEARCH / "state_windows.py",
        ["_BASE_FEATURE_COLS", "_ENTROPY_FEATURE_COLS", "_DELTA_SOURCE"])
    expect = (src["_BASE_FEATURE_COLS"] + src["_ENTROPY_FEATURE_COLS"]
              + [f"d_{c}" for c in src["_DELTA_SOURCE"]])
    assert STATE_FEATURE_COLS == expect, "STATE_FEATURE_COLS drifted"


def test_attack_table_matches_research():
    assert (_attack_table(_RESEARCH / "attack_stages.py")
            == _attack_table(_VENDOR / "attack_stages.py"))
