#!/usr/bin/env python3
# =============================================================================
# sentinel_infer | attack_stages.py
# VENDORED VERBATIM from research/sentinel_wm/attack_stages.py
# (only the config import changed: `from . import schema as C`). Keep in sync.
# =============================================================================
# MITRE ATT&CK phase mapping: a deterministic, auditable TWO-LAYER function
# applied AFTER the model. Layer A = static family -> tactic/technique table.
# Layer B = context resolver that lowers confidence when evidence is weak.
# Pure Python, no torch / pandas.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

from . import schema as C


@dataclass(frozen=True)
class AttackRef:
    tactic: str
    techniques: tuple
    kill_chain_phase: str
    base_confidence: str
    note: str = ""


_NONE = AttackRef("None", (), "None", "High", "all flows benign")

CIC_LABEL_TO_ATTACK: Dict[str, AttackRef] = {
    "BENIGN": _NONE,
    "PortScan": AttackRef(
        "Reconnaissance / Discovery", ("T1595", "T1046"),
        "Reconnaissance", "High",
        "sequential/randomised port sweep is an unambiguous recon signature"),
    "FTP-Patator": AttackRef(
        "Credential Access / Initial Access", ("T1110", "T1190"),
        "Initial Access", "Medium",
        "brute force against a service; tactic depends on success"),
    "SSH-Patator": AttackRef(
        "Credential Access / Initial Access", ("T1110", "T1021"),
        "Initial Access", "Medium",
        "brute force against SSH; Remote Services if it lands"),
    "Web Attack Brute Force": AttackRef(
        "Credential Access", ("T1110.004",),
        "Initial Access", "Medium", "credential stuffing over HTTP"),
    "Web Attack XSS": AttackRef(
        "Initial Access", ("T1190",),
        "Initial Access", "Medium", "exploit of a public-facing app"),
    "Web Attack SQL Injection": AttackRef(
        "Initial Access / Execution", ("T1190", "T1059"),
        "Initial Access", "Medium", "exploit + possible command execution"),
    "Heartbleed": AttackRef(
        "Initial Access / Credential Access", ("T1190",),
        "Initial Access", "Medium",
        "memory disclosure via a vulnerable TLS service"),
    "DoS Hulk": AttackRef(
        "Impact", ("T1499", "T1499.002"),
        "Impact", "High", "HTTP flood -> endpoint DoS"),
    "DoS GoldenEye": AttackRef(
        "Impact", ("T1499", "T1499.002"),
        "Impact", "High", "HTTP keep-alive flood -> endpoint DoS"),
    "DoS slowloris": AttackRef(
        "Impact", ("T1499", "T1499.003"),
        "Impact", "High", "partial-request exhaustion (low-and-slow)"),
    "DoS Slowhttptest": AttackRef(
        "Impact", ("T1499", "T1499.003"),
        "Impact", "High", "slow-body exhaustion (low-and-slow)"),
    "DDoS": AttackRef(
        "Impact", ("T1498",),
        "Impact", "High", "volumetric network DoS"),
    "Bot": AttackRef(
        "Command and Control", ("T1071", "T1572"),
        "Command & Control", "Medium", "beaconing to a C2 server"),
    "Infiltration": AttackRef(
        "Initial Access -> Discovery -> Lateral Movement",
        ("T1190", "T1046", "T1021"),
        "Lateral Movement", "Medium",
        "external compromise then internal scanning from the victim host"),
}

_RECON_LIKE = {"PortScan", "Infiltration"}
_PORT_AXIS = {"PortScan", "Infiltration"}
_CONF_ORDER = {"Low": 0, "Medium": 1, "High": 2}
_CONF_INV = {v: k for k, v in _CONF_ORDER.items()}


def _downgrade(conf: str, steps: int = 1) -> str:
    return _CONF_INV[max(0, _CONF_ORDER.get(conf, 1) - steps)]


@dataclass
class StageAssessment:
    progression_state: str
    mitre_tactic: str
    technique_ids: List[str]
    kill_chain_phase: str
    confidence: str
    rationale: str
    dominant_family: str = "BENIGN"
    family_transition: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["technique_ids"] = list(self.technique_ids)
        return d


def assess_window(progression_state: str,
                  dominant_family: str = "BENIGN",
                  prev_family: Optional[str] = None,
                  attack_ratio: float = 0.0) -> StageAssessment:
    ps = progression_state.upper()
    fam = dominant_family or "BENIGN"
    ref = CIC_LABEL_TO_ATTACK.get(fam, AttackRef(
        "Unknown", (), "Unknown", "Low", "family not in knowledge base"))

    if ps == "NORMAL":
        return StageAssessment(
            "NORMAL", "None", [], "None", "High",
            "window contains only BENIGN flows", "BENIGN")

    if ps == "PRE_ATTACK":
        if fam in _RECON_LIKE:
            return StageAssessment(
                "PRE_ATTACK", "Reconnaissance", ["T1595", "T1590"],
                "Reconnaissance",
                "Medium" if fam in _PORT_AXIS else "Low",
                f"benign-labelled window {C.CONFIG.window.pre_attack_span} steps "
                f"before a confirmed {fam} onset; scan-shaped precursor", fam)
        return StageAssessment(
            "PRE_ATTACK", "Reconnaissance (inferred)", ["T1590"],
            "Reconnaissance", "Low",
            f"benign window immediately preceding a {fam} onset; no confirmed "
            f"malicious flow yet - staging/ramp-up inferred from timing only",
            fam)

    if ps == "ONSET":
        return StageAssessment(
            "ONSET", ref.tactic, list(ref.techniques),
            ref.kill_chain_phase, _downgrade(ref.base_confidence, 1),
            f"first window carrying a confirmed {fam} label; tactic per "
            f"knowledge base but downgraded - single-window evidence "
            f"(attack_ratio={attack_ratio:.2f})", fam)

    if ps == "ACTIVE":
        conf = ref.base_confidence
        if attack_ratio < 0.15:
            conf = _downgrade(conf, 1)
        return StageAssessment(
            "ACTIVE", ref.tactic, list(ref.techniques),
            ref.kill_chain_phase, conf,
            f"sustained {fam} activity (attack_ratio={attack_ratio:.2f}); "
            f"{ref.note}", fam)

    if ps == "CONTINUATION":
        if prev_family and prev_family != fam and fam != "BENIGN":
            new_ref = CIC_LABEL_TO_ATTACK.get(fam, ref)
            return StageAssessment(
                "CONTINUATION", new_ref.tactic, list(new_ref.techniques),
                new_ref.kill_chain_phase, _downgrade(new_ref.base_confidence, 1),
                f"tactic transition {prev_family} -> {fam}; adversary changed "
                f"technique within the same episode", fam,
                family_transition=f"{prev_family} -> {fam}")
        return StageAssessment(
            "CONTINUATION", ref.tactic or "Impact", list(ref.techniques),
            ref.kill_chain_phase, _downgrade(ref.base_confidence, 1),
            f"{fam} episode tail (attack_ratio={attack_ratio:.2f}); activity "
            f"persisting or winding down", fam)

    return StageAssessment(
        ps, ref.tactic, list(ref.techniques), ref.kill_chain_phase,
        "Low", f"unrecognised progression state '{progression_state}'", fam)


def assess_forecast(state_probs: Sequence[float],
                    horizon_k: int,
                    dominant_family_hint: str = "BENIGN",
                    prev_family: Optional[str] = None,
                    attack_prob: float = 0.0) -> StageAssessment:
    idx = int(max(range(len(state_probs)), key=lambda i: state_probs[i]))
    ps = C.IDX_TO_STATE[idx]
    ratio_proxy = float(attack_prob)
    fam = dominant_family_hint or "BENIGN"
    a = assess_window(ps, fam, prev_family, ratio_proxy)

    if ps in ("ONSET", "ACTIVE", "CONTINUATION") and fam == "BENIGN":
        a.mitre_tactic = "Unspecified (family unknown at forecast time)"
        a.technique_ids = []
        a.kill_chain_phase = "Attack (unspecified)"
        a.confidence = "Low"

    steps_down = 1 + (max(0, horizon_k - 1) // 3)
    a.confidence = _downgrade(a.confidence, steps_down)
    a.rationale = (f"forecast +{horizon_k * C.CONFIG.window.window_seconds}s "
                   f"(step k={horizon_k}); P(state)={state_probs[idx]:.2f}, "
                   f"P(attack)={attack_prob:.2f}. " + a.rationale)
    return a


def label_frame_stages(prog_states: Sequence[str],
                       dominant_families: Sequence[str],
                       attack_ratios: Sequence[float]) -> List[dict]:
    out, prev_fam = [], None
    for ps, fam, ratio in zip(prog_states, dominant_families, attack_ratios):
        a = assess_window(ps, fam if fam else "BENIGN", prev_fam, float(ratio))
        out.append(a.to_dict())
        if str(ps).upper() in ("ONSET", "ACTIVE", "CONTINUATION"):
            prev_fam = fam if fam else prev_fam
        elif str(ps).upper() == "NORMAL":
            prev_fam = None
    return out
