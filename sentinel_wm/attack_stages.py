#!/usr/bin/env python3
# =============================================================================
# SENTINEL-WM  |  attack_stages.py
# -----------------------------------------------------------------------------
# MITRE ATT&CK phase mapping  --  the part the proposal is most careful about
# (sections 4/Innovation-4, 6.7, 8, 11.4) and the part reviewers attack first.
#
# THE HONEST CONSTRAINT
# --------------------
# CIC-IDS-2017 gives you an *attack-type label per flow* ("this flow is
# DoS Hulk"). It does NOT give you a *MITRE tactic label per time window*
# ("at 11:42:30 the adversary is in the Impact tactic"). So we NEVER train a
# model directly on ATT&CK tactics. Instead the mapping is a deterministic,
# auditable, TWO-LAYER function applied *after* the model has produced its
# data-driven output:
#
#   LAYER A  (static knowledge base) :  attack_family  -> ATT&CK tactic(s),
#                                       technique IDs, base confidence.
#            This is literature, not learned.  See CIC_LABEL_TO_ATTACK below.
#
#   LAYER B  (context resolver)      :  (progression_state, dominant_family,
#                                       family_transition, attack_ratio)
#                                       -> a single ATT&CK phase label with a
#                                       confidence that is *lowered* whenever
#                                       the evidence is weak (PRE_ATTACK,
#                                       ONSET, low ratio, forecast horizon).
#
# Every output carries an explicit `confidence` in {High, Medium, Low} and a
# short `rationale` string, so a defender (or a judge) can see exactly why the
# label was assigned. The system never emits an ATT&CK phase as if it were
# ground truth.
#
# Nothing here imports torch / pandas - it is pure Python and unit-testable.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

from sentinel_wm import config as C


# -----------------------------------------------------------------------------
# LAYER A  -- static ATT&CK knowledge base
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AttackRef:
    tactic: str                 # ATT&CK tactic name (or "None")
    techniques: tuple           # ATT&CK technique IDs
    kill_chain_phase: str       # coarse phase used on the dashboard timeline
    base_confidence: str        # High | Medium | Low  (per proposal table 8.2)
    note: str = ""


_NONE = AttackRef("None", (), "None", "High", "all flows benign")

# Keys are canonical `attack_family` values from preprocessing.normalise_label().
CIC_LABEL_TO_ATTACK: Dict[str, AttackRef] = {
    "BENIGN": _NONE,

    # ---- Reconnaissance / Discovery -------------------------------------- --
    "PortScan": AttackRef(
        "Reconnaissance / Discovery", ("T1595", "T1046"),
        "Reconnaissance", "High",
        "sequential/randomised port sweep is an unambiguous recon signature"),

    # ---- Credential Access / Initial Access ---------------------------------
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

    # ---- Impact  (all CIC-IDS-2017 DoS/DDoS families) ----------------------
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

    # ---- Command & Control ---------------------------------------------- --
    "Bot": AttackRef(
        "Command and Control", ("T1071", "T1572"),
        "Command & Control", "Medium", "beaconing to a C2 server"),

    # ---- Multi-phase ----------------------------------------------------- --
    "Infiltration": AttackRef(
        "Initial Access -> Discovery -> Lateral Movement",
        ("T1190", "T1046", "T1021"),
        "Lateral Movement", "Medium",
        "external compromise then internal scanning from the victim host"),
}

# Families whose *pre-attack* window is genuinely reconnaissance-shaped.
_RECON_LIKE = {"PortScan", "Infiltration"}
# Families that scan on a *port* axis (drives which Tier-2 features matter).
_PORT_AXIS = {"PortScan", "Infiltration"}

_CONF_ORDER = {"Low": 0, "Medium": 1, "High": 2}
_CONF_INV = {v: k for k, v in _CONF_ORDER.items()}


def _downgrade(conf: str, steps: int = 1) -> str:
    return _CONF_INV[max(0, _CONF_ORDER.get(conf, 1) - steps)]


# -----------------------------------------------------------------------------
# LAYER B  -- context resolver
# -----------------------------------------------------------------------------
@dataclass
class StageAssessment:
    progression_state: str          # NORMAL / PRE_ATTACK / ONSET / ACTIVE / CONT.
    mitre_tactic: str
    technique_ids: List[str]
    kill_chain_phase: str           # coarse phase for the timeline
    confidence: str                 # High | Medium | Low
    rationale: str
    dominant_family: str = "BENIGN"
    family_transition: Optional[str] = None   # "DoS Hulk -> DoS GoldenEye"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["technique_ids"] = list(self.technique_ids)
        return d


def assess_window(progression_state: str,
                  dominant_family: str = "BENIGN",
                  prev_family: Optional[str] = None,
                  attack_ratio: float = 0.0) -> StageAssessment:
    """
    Resolve a single state window to an ATT&CK phase.

    Parameters
    ----------
    progression_state : one of config.PROGRESSION_STATES  (model's primary target)
    dominant_family   : most frequent non-benign `attack_family` in the window
                        (or "BENIGN")
    prev_family       : dominant family of the previous attack window, if any
    attack_ratio      : fraction of flows in the window that are malicious
    """
    ps = progression_state.upper()
    fam = dominant_family or "BENIGN"
    ref = CIC_LABEL_TO_ATTACK.get(fam, AttackRef(
        "Unknown", (), "Unknown", "Low", "family not in knowledge base"))

    # ---------- NORMAL ----------------------------------------------------- -
    if ps == "NORMAL":
        return StageAssessment(
            "NORMAL", "None", [], "None", "High",
            "window contains only BENIGN flows", "BENIGN")

    # ---------- PRE_ATTACK  (benign traffic, but temporally adjacent to an
    #            onset -> forecasting target, weak evidence) ----------------
    if ps == "PRE_ATTACK":
        if fam in _RECON_LIKE:
            return StageAssessment(
                "PRE_ATTACK", "Reconnaissance", ["T1595", "T1590"],
                "Reconnaissance",
                "Medium" if fam in _PORT_AXIS else "Low",
                f"benign-labelled window {C.CONFIG.window.pre_attack_span} steps "
                f"before a confirmed {fam} onset; scan-shaped precursor", fam)
        # generic precursor to a non-recon family (e.g. DoS ramp-up)
        return StageAssessment(
            "PRE_ATTACK", "Reconnaissance (inferred)", ["T1590"],
            "Reconnaissance", "Low",
            f"benign window immediately preceding a {fam} onset; no confirmed "
            f"malicious flow yet - staging/ramp-up inferred from timing only",
            fam)

    # ---------- ONSET  (first confirmed attack window - still ambiguous) ---
    if ps == "ONSET":
        return StageAssessment(
            "ONSET", ref.tactic, list(ref.techniques),
            ref.kill_chain_phase, _downgrade(ref.base_confidence, 1),
            f"first window carrying a confirmed {fam} label; tactic per "
            f"knowledge base but downgraded - single-window evidence "
            f"(attack_ratio={attack_ratio:.2f})", fam)

    # ---------- ACTIVE  (sustained - strongest evidence) -----------------
    if ps == "ACTIVE":
        conf = ref.base_confidence
        if attack_ratio < 0.15:
            conf = _downgrade(conf, 1)
        return StageAssessment(
            "ACTIVE", ref.tactic, list(ref.techniques),
            ref.kill_chain_phase, conf,
            f"sustained {fam} activity (attack_ratio={attack_ratio:.2f}); "
            f"{ref.note}", fam)

    # ---------- CONTINUATION  (attack persists; family may have shifted) --
    if ps == "CONTINUATION":
        if prev_family and prev_family != fam and fam != "BENIGN":
            new_ref = CIC_LABEL_TO_ATTACK.get(fam, ref)
            return StageAssessment(
                "CONTINUATION", new_ref.tactic, list(new_ref.techniques),
                new_ref.kill_chain_phase, _downgrade(new_ref.base_confidence, 1),
                f"tactic transition {prev_family} -> {fam}; adversary changed "
                f"technique within the same episode", fam,
                family_transition=f"{prev_family} -> {fam}")
        # same family, trailing edge of the episode
        return StageAssessment(
            "CONTINUATION", ref.tactic or "Impact", list(ref.techniques),
            ref.kill_chain_phase, _downgrade(ref.base_confidence, 1),
            f"{fam} episode tail (attack_ratio={attack_ratio:.2f}); activity "
            f"persisting or winding down", fam)

    # ---------- fallback -------------------------------------------------- --
    return StageAssessment(
        ps, ref.tactic, list(ref.techniques), ref.kill_chain_phase,
        "Low", f"unrecognised progression state '{progression_state}'", fam)


# -----------------------------------------------------------------------------
# Forecast variant  -- map a K-step rollout to an ATT&CK phase
# -----------------------------------------------------------------------------
def assess_forecast(state_probs: Sequence[float],
                    horizon_k: int,
                    dominant_family_hint: str = "BENIGN",
                    prev_family: Optional[str] = None,
                    attack_prob: float = 0.0) -> StageAssessment:
    """
    Turn the progression-head distribution at horizon step k into an ATT&CK
    assessment. Confidence is lowered by one notch for every 3 horizon steps
    (rollout error accumulates - proposal 6.6 / risk table).

    state_probs : length-5 vector over config.PROGRESSION_STATES
    horizon_k   : 1..K  (which forecast step this is)
    """
    idx = int(max(range(len(state_probs)), key=lambda i: state_probs[i]))
    ps = C.IDX_TO_STATE[idx]
    ratio_proxy = float(attack_prob)
    fam = dominant_family_hint or "BENIGN"
    a = assess_window(ps, fam, prev_family, ratio_proxy)

    # forecast says "attack" but we have no family context from the present -
    # do not silently emit phase "None"; say so explicitly.
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


# -----------------------------------------------------------------------------
# Vectorised helper for labelling a whole state_windows frame
# -----------------------------------------------------------------------------
def label_frame_stages(prog_states: Sequence[str],
                       dominant_families: Sequence[str],
                       attack_ratios: Sequence[float]) -> List[dict]:
    """Row-wise assess_window over aligned sequences -> list of dicts."""
    out, prev_fam = [], None
    for ps, fam, ratio in zip(prog_states, dominant_families, attack_ratios):
        a = assess_window(ps, fam if fam else "BENIGN", prev_fam, float(ratio))
        out.append(a.to_dict())
        if str(ps).upper() in ("ONSET", "ACTIVE", "CONTINUATION"):
            prev_fam = fam if fam else prev_fam
        elif str(ps).upper() == "NORMAL":
            prev_fam = None
    return out


# -----------------------------------------------------------------------------
# self-test
# -----------------------------------------------------------------------------
def _self_test() -> bool:
    ok = True
    a = assess_window("NORMAL")
    ok &= a.kill_chain_phase == "None" and a.confidence == "High"

    a = assess_window("ACTIVE", "DoS Hulk", attack_ratio=0.8)
    ok &= a.kill_chain_phase == "Impact" and "T1499" in a.technique_ids
    ok &= a.confidence == "High"

    a = assess_window("ONSET", "DoS Hulk", attack_ratio=0.2)
    ok &= a.confidence == "Medium"          # High downgraded once

    a = assess_window("PRE_ATTACK", "PortScan")
    ok &= a.kill_chain_phase == "Reconnaissance" and a.confidence == "Medium"

    a = assess_window("PRE_ATTACK", "DoS Hulk")
    ok &= a.confidence == "Low"             # inferred precursor

    a = assess_window("CONTINUATION", "DoS GoldenEye", prev_family="DoS Hulk")
    ok &= a.family_transition == "DoS Hulk -> DoS GoldenEye"

    a = assess_forecast([0.1, 0.1, 0.1, 0.6, 0.1], horizon_k=6,
                        dominant_family_hint="DoS Hulk", attack_prob=0.7)
    ok &= a.confidence == "Low"             # High - 3 notches, floored at Low
    print("attack_stages self-test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
