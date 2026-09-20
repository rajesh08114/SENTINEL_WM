"""Rule engine evaluating deterministic security and heuristic policies alongside AI forecasts."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from .models import RuleDefinition, RuleMatch, RuleSeverity


class Rule:
    def __init__(
        self,
        rule_id: str,
        name: str,
        severity: RuleSeverity,
        description: str,
        mitre_technique: str,
        action: str,
        predicate: Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str]],
        enabled: bool = True,
    ):
        self.rule_id = rule_id
        self.name = name
        self.severity = severity
        self.description = description
        self.mitre_technique = mitre_technique
        self.action = action
        self.predicate = predicate
        self.enabled = enabled

    def to_definition(self) -> RuleDefinition:
        return RuleDefinition(
            id=self.rule_id,
            name=self.name,
            severity=self.severity.value,
            description=self.description,
            mitre_technique=self.mitre_technique,
            action=self.action,
            enabled=self.enabled,
        )


class RuleEngine:
    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._init_default_rules()

    def register_rule(self, rule: Rule) -> None:
        self._rules[rule.rule_id] = rule

    def get_definitions(self) -> List[RuleDefinition]:
        return [r.to_definition() for r in self._rules.values()]

    def set_enabled(self, rule_id: str, enabled: bool) -> bool:
        if rule_id in self._rules:
            self._rules[rule_id].enabled = enabled
            return True
        return False

    def evaluate_anchor(
        self,
        anchor: Dict[str, Any],
        window_features: Optional[Dict[str, Any]] = None,
    ) -> List[RuleMatch]:
        """Evaluate all active rules against an anchor forecast and its window features."""
        matches: List[RuleMatch] = []
        features = window_features or {}

        for rule in self._rules.values():
            if not rule.enabled:
                continue
            try:
                matched, detail = rule.predicate(features, anchor)
                if matched:
                    matches.append(
                        RuleMatch(
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            severity=rule.severity.value,
                            description=rule.description,
                            mitre_technique=rule.mitre_technique,
                            action=rule.action,
                            matched_value=detail,
                        )
                    )
            except Exception:
                continue

        return matches

    def _init_default_rules(self) -> None:
        # Rule 1: TCP SYN / Port Scan heuristic
        def check_syn_scan(feat: Dict[str, Any], anch: Dict[str, Any]) -> Tuple[bool, str]:
            syn_r = float(feat.get("syn_ratio", 0.0) or 0.0)
            fl_cnt = float(feat.get("flow_count", 0.0) or 0.0)
            if syn_r >= 0.70 and fl_cnt >= 25:
                return True, f"SYN flag ratio {syn_r:.1%} across {int(fl_cnt)} flows in window"
            return False, ""

        self.register_rule(
            Rule(
                rule_id="RULE-SEC-001",
                name="Aggressive TCP SYN Scan",
                severity=RuleSeverity.HIGH,
                description="High concentration of SYN packets without ACK completion, characteristic of reconnaissance scans.",
                mitre_technique="T1046: Network Service Discovery",
                action="Apply source IP rate-limiting and verify against external threat intel feeds.",
                predicate=check_syn_scan,
            )
        )

        # Rule 2: Destination Port Entropy / Port Sweep
        def check_port_sweep(feat: Dict[str, Any], anch: Dict[str, Any]) -> Tuple[bool, str]:
            port_ent = float(feat.get("port_entropy", 0.0) or feat.get("dst_port_entropy", 0.0) or 0.0)
            fl_cnt = float(feat.get("flow_count", 0.0) or 0.0)
            if port_ent >= 0.85 and fl_cnt >= 20:
                return True, f"Normalized port distribution entropy {port_ent:.2f} (threshold: 0.85)"
            return False, ""

        self.register_rule(
            Rule(
                rule_id="RULE-SEC-002",
                name="Network Port Sweep",
                severity=RuleSeverity.MEDIUM,
                description="Unusually high entropy across target ports indicating horizontal or vertical port scanning.",
                mitre_technique="T1046: Network Service Discovery",
                action="Review targeted port range in firewall logs; trigger temporary port block.",
                predicate=check_port_sweep,
            )
        )

        # Rule 3: High Outbound Velocity / Potential Exfiltration
        def check_exfil_burst(feat: Dict[str, Any], anch: Dict[str, Any]) -> Tuple[bool, str]:
            bytes_sec = float(feat.get("bytes_per_sec", 0.0) or feat.get("fwd_bytes_rate", 0.0) or 0.0)
            pkt_rate = float(feat.get("packet_rate", 0.0) or feat.get("fwd_pkt_rate", 0.0) or 0.0)
            if bytes_sec >= 5_000_000 or (bytes_sec >= 2_000_000 and pkt_rate >= 1000):
                mb_s = bytes_sec / 1_000_000
                return True, f"Outbound throughput spiked to {mb_s:.2f} MB/s"
            return False, ""

        self.register_rule(
            Rule(
                rule_id="RULE-SEC-003",
                name="High-Volume Data Velocity",
                severity=RuleSeverity.HIGH,
                description="Abrupt outbound byte transfer surge exceeding typical operational baseline.",
                mitre_technique="T1048: Exfiltration Over Alternative Protocol",
                action="Isolate egress socket and inspect destination IP geolocations.",
                predicate=check_exfil_burst,
            )
        )

        # Rule 4: Attack Onset Early Warning
        def check_ai_onset(feat: Dict[str, Any], anch: Dict[str, Any]) -> Tuple[bool, str]:
            horizon = anch.get("horizon", [])
            for h in horizon:
                state = h.get("progression_state", "")
                p = float(h.get("attack_prob", 0.0) or 0.0)
                if state in ("ONSET", "ACTIVE") and p >= 0.70:
                    k = h.get("k", 1)
                    secs = h.get("horizon_seconds", k * 10)
                    return True, f"Predicted {state} at +{secs}s with P(attack)={p:.1%}"
            return False, ""

        self.register_rule(
            Rule(
                rule_id="RULE-AI-004",
                name="Predictive Attack Onset Trigger",
                severity=RuleSeverity.CRITICAL,
                description="World Model forward trajectory projects attack onset before completion of adversary kill chain.",
                mitre_technique="T1190: Exploit Public-Facing Application",
                action="Engage Incident Response Playbook and stage automated endpoint isolation.",
                predicate=check_ai_onset,
            )
        )

        # Rule 5: Immediate Lead Time Alarm
        def check_short_lead_time(feat: Dict[str, Any], anch: Dict[str, Any]) -> Tuple[bool, str]:
            lt = anch.get("lead_time_seconds", 0)
            prob = max(anch.get("max_detection_prob") or 0.0, anch.get("max_attack_prob") or 0.0)
            if anch.get("alert") and 0 < lt <= 20 and prob >= 0.85:
                return True, f"Lead time {lt}s remaining with high-confidence probability {prob:.1%}"
            return False, ""

        self.register_rule(
            Rule(
                rule_id="RULE-AI-005",
                name="Imminent Critical Trajectory",
                severity=RuleSeverity.CRITICAL,
                description="Alert triggered within immediate horizon (<= 20s lead time) with high model certainty.",
                mitre_technique="T1499: Endpoint DoS",
                action="Execute automated defensive containment.",
                predicate=check_short_lead_time,
            )
        )


RULE_ENGINE = RuleEngine()
