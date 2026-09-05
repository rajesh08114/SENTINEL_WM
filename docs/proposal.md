
**Proposal for AI-Based Network Attack Forecasting from Network Traffic Data**
**Version:** 2.0 — Reviewed & Validated
**Classification:** Open-Source Research Prototype
**Domain:** Proactive Cyber Defence | Predictive Threat Intelligence | Enterprise & Critical Infrastructure Security

---

> **Version 2.0 Change Summary**
> This document incorporates a full technical review against the actual CIC-IDS-2017 dataset structure, attack scenarios, and feature feasibility. Key changes: (1) primary dataset corrected to CIC-IDS-2017 with accurate attack day schedule; (2) feature schema corrected and formally tiered into Flow / Packet / Behaviour / Metadata layers; (3) MITRE ATT&CK mapping repositioned as a secondary semantic interpretation layer, not primary ground truth; (4) MITRE "attractor embeddings" redesigned as ATT&CK-aligned phase interpretation — more defensible and implementation-honest; (5) raw IP addresses removed as model features, replaced with behavioural derivatives; (6) source_file and timestamp removed as model inputs; (7) train/val/test split corrected from random-row to day-based; (8) Infiltration scenario elevated as the primary demonstration case; (9) evaluation extended with Lead Time, Forecast Horizon Accuracy, and Calibration metrics; (10) three-tier build strategy (Baseline → MVP → Advanced) introduced to ensure deliverability; (11) 10-second window recommendation adopted with 6-step horizon.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Understanding — The Defender's Dilemma](#2-problem-understanding--the-defenders-dilemma)
3. [Why Existing Approaches Fail](#3-why-existing-approaches-fail)
4. [Innovation Thesis](#4-innovation-thesis)
5. [Solution Overview — SENTINEL-WM Architecture](#5-solution-overview--sentinel-wm-architecture)
6. [Detailed Architecture](#6-detailed-architecture)
   - 6.1 Data Ingestion & Source Strategy
   - 6.2 Feature Extraction Pipeline (Corrected & Tiered)
   - 6.3 Network State Construction
   - 6.4 Graph-Temporal State Encoder
   - 6.5 Temporal World Model Core
   - 6.6 Forward Simulation Engine
   - 6.7 Attack Progression State & ATT&CK-Aligned Interpretation
   - 6.8 Explainability Engine
   - 6.9 Defender Dashboard
7. [Data Strategy](#7-data-strategy)
8. [MITRE ATT&CK Integration — Corrected Approach](#8-mitre-attck-integration--corrected-approach)
9. [System Workflows](#9-system-workflows)
10. [Evaluation & Benchmarking Strategy](#10-evaluation--benchmarking-strategy)
11. [Key Innovations & Justifications](#11-key-innovations--justifications)
12. [Three-Tier Build Strategy](#12-three-tier-build-strategy)
13. [Technology Stack](#13-technology-stack)
14. [Deployment Architecture](#14-deployment-architecture)
15. [Risk Analysis & Mitigations](#15-risk-analysis--mitigations)
16. [Appendix: Feature Reference Tables](#16-appendix-feature-reference-tables)

---

## 1. Executive Summary

**SENTINEL-WM** (Spatio-Temporal Enemy Network Intelligence with Learned — World Model) is a proactive cyber defence system that abandons reactive per-flow classification in favour of **temporal forward simulation of attack progressions**.

Rather than labelling individual flows as benign or malicious, SENTINEL-WM aggregates traffic into 10-second **network state windows**, learns how those states evolve over time, and **rolls the world forward K steps** to estimate the probability that the current trajectory converges to a compromise state before the attacker completes their kill chain.

The system targets the **detection-to-action gap** — the window between first signs of adversarial behaviour and final compromise. By predicting where an attack is headed, not just where it has been, defenders gain **actionable lead time** measured in tens of seconds to minutes: enough time to isolate assets, revoke credentials, or invoke pre-built IR playbooks before damage is done.

**Core differentiators at a glance:**

| Capability | Traditional ML IDS | SENTINEL-WM |
|---|---|---|
| Unit of analysis | Single flow | Evolving 10-second network state window |
| Temporal modelling | None or sliding window | Learned state-transition dynamics |
| Output | Binary label | K-step attack probability + progression state |
| ATT&CK awareness | Post-hoc heuristics | ATT&CK-aligned phase interpretation layer |
| Explainability | Static feature importance | Attention + SHAP + temporal saliency |
| Action horizon | Past (what happened) | Future (what will happen in next K×10s) |
| Lead time metric | Not reported | Mean Lead Time explicitly measured |

---

## 2. Problem Understanding — The Defender's Dilemma

> *"An attacker needs to succeed once. A defender needs to succeed every time."*
> — Classic asymmetry of cyber operations

After 10 years working across SOC floors, red team engagements, and CIRT incident response, the pattern is always the same: **by the time the alert fires, the game is already half over.**

### The Kill Chain Has Temporal Structure

An infiltration is not a single event — it is a **narrative unfolding over time**:

```
[Reconnaissance]  →  [Initial Access]
        ↓
[Persistence & Privilege Escalation]
        ↓
[Lateral Movement]  →  [Discovery]
        ↓
[Command & Control]  →  [Exfiltration / Impact]
```

Each phase generates **distinctive traffic signatures that evolve causally from the previous phase**. A slow SYN scan at T=0 is not a random isolated event — it causally influences which ports receive connection attempts at T+1, which services are exploited at T+2, and which hosts begin beaconing at T+3.

### What Traditional Systems Miss

Current NIDS operate on individual flows or short fixed windows. They answer: *"Is this flow anomalous?"* — not *"Given everything I have seen in the last 10 windows, is this network on a trajectory toward compromise in the next 6?"*

This means:
- **Slow-burn attacks** (APTs spreading recon over minutes to hours) evade per-flow anomaly detectors
- **Multi-stage correlation** is left entirely to human analysts who are already overwhelmed
- **No lead time** — the alert arrives concurrent with or after the compromise event
- **Alert fatigue** — without temporal context, every alert is a flat signal with no priority gradient

### The Real Cost

In enterprise environments and Critical Information Infrastructure (CII), the cost of this gap is catastrophic. SCADA/ICS environments are particularly vulnerable — a 40-second lead time before a compromised host begins internal scanning could mean the difference between a contained incident and a full-network breach.

---

## 3. Why Existing Approaches Fail

### 3.1 Static Classifiers (Random Forest, XGBoost on CIC-IDS features)

```
Problem: Each flow is an i.i.d. sample. Temporal causality is discarded.
Result:  Works on per-flow signatures. Fails on slow reconnaissance,
         polymorphic attacks, and low-and-slow exfiltration.
```

### 3.2 LSTM-Based Anomaly Detection

```
Problem: Learns to predict "next normal" but cannot simulate
         adversarial progression across states.
Result:  Good at flagging deviations from baseline. Poor at
         forecasting multi-step attack progressions. No stage mapping.
```

### 3.3 Static Graph-Based Methods (GNN, single-snapshot)

```
Problem: Typically single-snapshot. No temporal dynamics.
Result:  Captures lateral movement topology but misses the
         time-sequenced evolution of attack stages.
```

### 3.4 Rule-Based SIEM Correlation

```
Problem: Rules require human authorship. Cannot generalise to
         novel attack patterns or zero-days.
Result:  High false-positive rate, brittle to obfuscation,
         no probabilistic scoring, no forward projection.
```

**The gap:** No existing deployed system combines **(a) temporal state modelling from aggregated windows**, **(b) K-step forward simulation**, **(c) calibrated attack probability with confidence intervals**, and **(d) interpretable ATT&CK-aligned stage prediction** into a unified architecture with a rigorously measured lead time advantage.

---

## 4. Innovation Thesis

SENTINEL-WM is built on four interconnected innovations:

### Innovation 1 — Dual-Source, Tiered Feature Architecture

Network state is constructed from **two authoritative sources** used for their respective strengths:
- **CIC-IDS-2017 CSV** → CICFlowMeter-derived flow-level features (authoritative for aggregate flow behaviour)
- **CIC-IDS-2017 PCAP** → Scapy-derived packet-level features (authoritative for per-packet micro-signatures)

Features are formally tiered: Flow → Packet → Behaviour → Metadata. Behavioural derivatives (port scan entropy, fan-out, burstiness) are computed from the above two layers and kept explicitly distinct from raw measurements.

### Innovation 2 — 10-Second Network State Windows as the Core Object

Rather than training on flows, SENTINEL-WM trains on **structured network state vectors S_t**, each summarising 10 seconds of traffic from both PCAP and CSV sources. This transforms the problem from flow classification to **state trajectory modelling** — a fundamentally richer representation that preserves temporal causality.

### Innovation 3 — Temporal World Model with K-Step Forward Simulation

A Temporal Transformer learns P(S_{t+1} | S_{t-L:t}) — how network state evolves given recent history. At inference, it performs **autoregressive rollout** over K=6 steps (60 seconds ahead), producing a time-series attack probability score with confidence intervals. This enables the system to warn of an impending attack stage before it fully materialises.

### Innovation 4 — ATT&CK-Aligned Phase Interpretation (Not Hard-Coded Mapping)

Rather than forcing CIC-IDS-2017 labels into ATT&CK phases as primary ground truth (which the dataset cannot reliably support), SENTINEL-WM defines a set of **data-driven attack progression states** (NORMAL → PRE-ATTACK → ONSET → ACTIVE → CONTINUATION) learned from the dataset's temporal annotations. These are then **secondarily interpreted** through ATT&CK-aligned semantic labels with explicitly stated confidence, making the mapping defensible under scrutiny.

---

## 5. Solution Overview — SENTINEL-WM Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     SENTINEL-WM: HIGH-LEVEL ARCHITECTURE v2.0                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐   ║
║  │                        DUAL-SOURCE INGESTION                            │   ║
║  │                                                                         │   ║
║  │   CIC-IDS-2017 PCAP              CIC-IDS-2017 CSV (CICFlowMeter)       │   ║
║  │   (Packet-level authority)        (Flow-level authority)                │   ║
║  └──────────┬──────────────────────────────────┬──────────────────────────┘   ║
║             │                                  │                               ║
║  ┌──────────▼──────────┐           ┌───────────▼──────────────────────────┐   ║
║  │  Packet Feature     │           │  Flow Feature Extraction             │   ║
║  │  Extraction (Scapy) │           │  (44-column CICFlowMeter subset)     │   ║
║  │  16 locked features │           │  Identity → Volume → Timing →        │   ║
║  └──────────┬──────────┘           │  Flags → Header/Window → Ratio       │   ║
║             │                      └───────────┬──────────────────────────┘   ║
║             │                                  │                               ║
║             └──────────────┬───────────────────┘                               ║
║                            │   5-tuple flow matching                           ║
║                            ▼   (src_ip, src_port, dst_ip, dst_port, proto +    ║
║  ┌─────────────────────────────────  flow timestamp)  ──────────────────────┐  ║
║  │              UNIFIED FLOW RECORDS (flow + packet features)               │  ║
║  └───────────────────────────────────┬──────────────────────────────────────┘  ║
║                                      │                                          ║
║  ┌───────────────────────────────────▼──────────────────────────────────────┐  ║
║  │              NETWORK STATE CONSTRUCTION  (10-second windows)             │  ║
║  │                                                                           │  ║
║  │   S_t = f(all flows + packets in window [t, t+10s])                      │  ║
║  │   → Temporal / Connection / TCP / Timing / Packet / Scanning /           │  ║
║  │     Behavioural summary vector (~60 dimensions)                          │  ║
║  │   → Behavioural graph G_t (nodes: hosts, edges: state per pair)          │  ║
║  │                                                                           │  ║
║  │   sequence: S_{t-9}, S_{t-8}, ..., S_{t-1}, S_t                         │  ║
║  └───────────────────────────────────┬──────────────────────────────────────┘  ║
║                                      │                                          ║
║  ┌───────────────────────────────────▼──────────────────────────────────────┐  ║
║  │                     TEMPORAL WORLD MODEL                                 │  ║
║  │                                                                           │  ║
║  │  [Optional: GAT spatial encoder per window → enriched h_t]               │  ║
║  │                                                                           │  ║
║  │  Temporal Transformer:                                                    │  ║
║  │  [S_{t-9} ... S_t] ──► z_t  (latent state, dim D)                       │  ║
║  │                                                                           │  ║
║  │  State Transition Head:                                                   │  ║
║  │  z_t ──► μ_{t+1}, σ_{t+1}  →  z_{t+1} ~ N(μ, σ)                        │  ║
║  │                                                                           │  ║
║  │  K-step autoregressive rollout: z_{t+1}, ..., z_{t+6}                    │  ║
║  └───────────────────────────────────┬──────────────────────────────────────┘  ║
║                                      │                                          ║
║  ┌───────────────────────────────────▼──────────────────────────────────────┐  ║
║  │               DUAL OUTPUT HEADS                                           │  ║
║  │                                                                           │  ║
║  │  Attack Head:         P(A_{t+k}=1 | S_{≤t})  for k=1..6                 │  ║
║  │                       → binary attack probability per future window       │  ║
║  │                                                                           │  ║
║  │  Progression Head:    P(Z_{t+k}=c | S_{≤t})  for each state c           │  ║
║  │                       → NORMAL / PRE-ATTACK / ONSET / ACTIVE / CONT.     │  ║
║  └───────────────────────────────────┬──────────────────────────────────────┘  ║
║                                      │                                          ║
║  ┌───────────────────────────────────▼──────────────────────────────────────┐  ║
║  │         ATT&CK-ALIGNED PHASE INTERPRETATION LAYER                        │  ║
║  │                                                                           │  ║
║  │  Progression state → ATT&CK tactic (secondary semantic mapping)          │  ║
║  │  Each mapping carries an explicit confidence label (High / Medium)       │  ║
║  │  (See §8 for full mapping table)                                          │  ║
║  └───────────────────────────────────┬──────────────────────────────────────┘  ║
║                                      │                                          ║
║  ┌───────────────────────────────────▼──────────────────────────────────────┐  ║
║  │                    EXPLAINABILITY ENGINE                                  │  ║
║  │                                                                           │  ║
║  │  • Attention weights → top contributing state windows                    │  ║
║  │  • SHAP values → top contributing state features                         │  ║
║  │  • Temporal saliency → gradient-based time-step importance               │  ║
║  └───────────────────────────────────┬──────────────────────────────────────┘  ║
║                                      │                                          ║
║  ┌───────────────────────────────────▼──────────────────────────────────────┐  ║
║  │                DEFENDER DASHBOARD (Streamlit — offline)                   │  ║
║  │                                                                           │  ║
║  │  • Attack probability timeline (K=6 steps, 60s horizon, with CI)        │  ║
║  │  • Progression state + ATT&CK-aligned interpretation                     │  ║
║  │  • Lead time counter since first crossing P(attack) > threshold          │  ║
║  │  • SHAP/attention explanation panel                                       │  ║
║  │  • Flagged state features and top contributing behaviours                │  ║
║  └───────────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 6. Detailed Architecture

### 6.1 Data Ingestion & Source Strategy

**Critical design principle:** PCAP and CSV serve different and complementary roles. They must not be conflated.

```
┌─────────────────────────────────────────────────────────────────────┐
│               DUAL-SOURCE STRATEGY                                  │
│                                                                     │
│  CIC-IDS-2017 CSV  (CICFlowMeter-generated)                        │
│  ├── Authoritative for: flow-level aggregate behaviour              │
│  ├── Contains: 78+ numeric features per bidirectional flow          │
│  ├── Loaded via: pandas / polars                                    │
│  └── Used as: primary flow feature source                          │
│                                                                     │
│  CIC-IDS-2017 PCAP (raw captures, ~70GB across 5 days)            │
│  ├── Authoritative for: packet-level micro-signatures              │
│  ├── Contains: individual packet headers and payloads              │
│  ├── Parsed via: Scapy + custom session aggregator                 │
│  └── Used as: packet feature source                                │
│                                                                     │
│  Flow Association (joining the two):                                │
│  ├── Join key: 5-tuple + flow start timestamp window               │
│  │   (src_ip, src_port, dst_ip, dst_port, proto, t_start±δ)       │
│  ├── NOT: 5-tuple alone (same tuple reoccurs across time)          │
│  └── Output: unified record per flow with both feature sets        │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 6.2 Feature Extraction Pipeline (Corrected & Tiered)

Features are formally organised into four tiers. This tiering is explicit in the codebase and documentation to make the feature pipeline reviewable and testable.

```
TIER 1 — FLOW FEATURES          (source: CIC-IDS-2017 CSV)
TIER 2 — PACKET FEATURES        (source: CIC-IDS-2017 PCAP via Scapy)
TIER 3 — BEHAVIOURAL FEATURES   (derived: computed from Tier 1 + Tier 2 per window)
TIER 4 — METADATA               (lineage only: NOT fed into the model)
```

#### TIER 1 — Flow Features (CICFlowMeter-derived subset, 44 columns)

> **Correct description:** These are a selected 44-column subset of the CICFlowMeter output (which produces 78+ features). They are drawn from the official CIC-IDS-2017 CSV files. Describing them as "CICFlowMeter-equivalent" is technically accurate; describing them as "all 44 CICFlowMeter columns" overstates their coverage.

```
┌──────────────────────────────────────────────────────────────────────┐
│  FLOW FEATURE EXTRACTOR (from CIC-IDS-2017 CSV)                     │
│                                                                      │
│  IDENTITY GROUP (6 columns — used for graph construction only;      │
│  raw IPs are NOT fed directly into the neural network):             │
│   • Src IP    →  derived: src_host_id (subnet hash)                 │
│   • Src Port  →  kept as-is                                         │
│   • Dst IP    →  derived: dst_host_id (subnet hash)                 │
│   • Dst Port  →  kept as-is                                         │
│   • Protocol  →  one-hot encoded                                    │
│   • Timestamp →  used for ordering only; NOT a model input feature  │
│                                                                      │
│  VOLUME / DURATION GROUP (5 columns):                               │
│   • Flow Duration                                                   │
│   • Total Fwd Packets                                               │
│   • Total Backward Packets                                          │
│   • Total Length of Fwd Packets                                     │
│   • Total Length of Bwd Packets                                     │
│                                                                      │
│  PACKET LENGTH STATS GROUP (12 columns):                            │
│   • Fwd Packet Length Max / Min / Mean / Std                        │
│   • Bwd Packet Length Max / Min / Mean / Std                        │
│   • Min Packet Length / Max Packet Length                           │
│   • Packet Length Mean / Std / Variance                             │
│                                                                      │
│  RATE GROUP (4 columns):                                            │
│   • Flow Bytes/s                                                    │
│   • Flow Packets/s                                                  │
│   • Fwd Packets/s                                                   │
│   • Bwd Packets/s                                                   │
│                                                                      │
│  INTER-ARRIVAL TIME GROUP (13 columns):                             │
│   • Flow IAT Mean / Std / Max / Min / Variance                      │
│   • Fwd IAT Total / Mean / Std / Max / Min                          │
│   • Bwd IAT Total / Mean / Std / Max / Min                          │
│                                                                      │
│  TCP FLAGS GROUP (10 columns):                                      │
│   • Fwd PSH Flags / Bwd PSH Flags                                   │
│   • Fwd URG Flags / Bwd URG Flags                                   │
│   • FIN / SYN / RST / PSH / ACK / URG / CWE / ECE Flag Count       │
│                                                                      │
│  HEADER & WINDOW GROUP (6 columns):                                 │
│   • Fwd Header Length / Bwd Header Length                           │
│   • Init_Win_bytes_forward / Init_Win_bytes_backward                │
│   • min_seg_size_forward                                            │
│   • act_data_pkt_fwd                                                │
│                                                                      │
│  RATIO & ACTIVE/IDLE GROUP (9 columns):                             │
│   • Down/Up Ratio                                                   │
│   • Active Mean / Std / Max / Min                                   │
│   • Idle Mean / Std / Max / Min                                     │
│                                                                      │
│  Output: 44 numeric flow features per flow (identity used           │
│          separately for graph construction, not as model input)     │
└──────────────────────────────────────────────────────────────────────┘
```

#### TIER 2 — Packet Features (PCAP-derived via Scapy, 18 locked columns)

> These 18 columns are computed exclusively from PCAP. They are the locked specification for packet-level features. They must not be derived from the CSV.

```
┌──────────────────────────────────────────────────────────────────────┐
│  PACKET FEATURE EXTRACTOR (from CIC-IDS-2017 PCAP via Scapy)       │
│                                                                      │
│  NETWORK LAYER (TTL & Fragmentation):                               │
│   • ttl_mean           — mean TTL across session packets            │
│   • ttl_variance       — high variance signals multi-tool/OS recon  │
│   • frag_more_flag_present  — MF flag observed (boolean)            │
│   • frag_dont_flag_present  — DF flag observed (boolean)            │
│   • frag_max_offset    — largest IP fragment offset seen            │
│                                                                      │
│  TRANSPORT LAYER (TCP):                                             │
│   • tcp_window_mean    — mean TCP window size                       │
│   • tcp_window_std     — variance in window size                    │
│   • retransmission_count — TCP retransmission count per session     │
│                                                                      │
│  PAYLOAD LAYER:                                                     │
│   • payload_min        — minimum payload size                       │
│   • payload_max        — maximum payload size                       │
│   • payload_variance   — payload size variance                      │
│   • payload_skew       — payload distribution skewness              │
│   • payload_kurtosis   — payload distribution kurtosis              │
│   • payload_nonzero_ratio  — fraction of packets with payload > 0   │
│                                                                      │
│  SCANNING SIGNATURES:                                               │
│   • port_scan_entropy        — Shannon entropy of dst port accesses │
│   • unique_dst_ports_per_src — unique dst ports per src per window  │
│   • port_scan_max_sequential_run — longest run of sequential ports  │
│   • port_scan_sequential_ratio   — sequential vs total port ratio   │
│                                                                      │
│  Output: 18 packet features per flow/session                        │
└──────────────────────────────────────────────────────────────────────┘
```

#### TIER 3 — Behavioural Features (window-level derived, NOT raw measurements)

> These are computed per 10-second window across all flows in that window, not per-flow. They capture aggregate network behaviour that only becomes visible at window level.

```
┌──────────────────────────────────────────────────────────────────────┐
│  BEHAVIOURAL FEATURE COMPUTATION (per 10-second window)             │
│                                                                      │
│  CONNECTION DYNAMICS:                                               │
│   • unique_src_count       — distinct source IPs                    │
│   • unique_dst_count       — distinct destination IPs               │
│   • unique_pairs           — distinct src→dst pairs                 │
│   • new_connections        — pairs appearing for first time         │
│   • fan_out                — max unique dst per single src          │
│   • fan_in                 — max unique src per single dst          │
│   • internal_external_ratio — inbound vs outbound flow proportion   │
│   • failed_connection_ratio — RST rate / total connection attempts  │
│                                                                      │
│  TEMPORAL DYNAMICS:                                                 │
│   • packet_rate            — total packets per second in window     │
│   • byte_rate              — total bytes per second in window       │
│   • flow_count             — number of flows in window              │
│   • burstiness             — coefficient of variation of IAT        │
│                                                                      │
│  TCP FLAG RATES (window-aggregate):                                 │
│   • syn_rate               — SYN flags per second                  │
│   • ack_rate               — ACK flags per second                  │
│   • rst_rate               — RST flags per second                  │
│   • fin_rate               — FIN flags per second                  │
│   • retransmission_rate    — window-level retransmission rate       │
│                                                                      │
│  Output: ~20 behavioural features per state window S_t             │
└──────────────────────────────────────────────────────────────────────┘
```

#### TIER 4 — Metadata (lineage only, never fed into model)

```
┌──────────────────────────────────────────────────────────────────────┐
│  METADATA (data lineage only — EXCLUDED from all model inputs)      │
│                                                                      │
│  • timestamp_window  — window start timestamp (for sequencing and   │
│                        evaluation only; NOT a model feature;        │
│                        feeding raw timestamp causes day-of-week     │
│                        memorisation)                                │
│                                                                      │
│  • source_file       — which CIC-IDS-2017 day CSV/PCAP this        │
│                        flow came from (for split assignment,        │
│                        debugging, and data lineage only;           │
│                        feeding source_file leaks train/test         │
│                        split identity directly into the model)      │
│                                                                      │
│  Derived from timestamp (safe to use as model input):               │
│  • time_since_previous_window — elapsed seconds between windows     │
│  • window_index_in_day        — positional index within day         │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 6.3 Network State Construction

The **Network State S_t** is the central object of SENTINEL-WM. Everything upstream produces it; everything downstream consumes it.

```
┌──────────────────────────────────────────────────────────────────────┐
│              NETWORK STATE S_t  (10-second window)                  │
│                                                                      │
│  S_t = concat(                                                      │
│                                                                      │
│    TEMPORAL:                                                         │
│      flow_count, packet_count, byte_count,                          │
│      packet_rate, byte_rate                                         │
│                                                                      │
│    CONNECTION:                                                       │
│      unique_src, unique_dst, unique_pairs,                          │
│      new_connections, fan_out, fan_in,                              │
│      internal_external_ratio, failed_connection_ratio               │
│                                                                      │
│    TCP FLAGS (aggregate):                                           │
│      syn_rate, ack_rate, rst_rate, fin_rate,                        │
│      retransmission_rate                                            │
│                                                                      │
│    TIMING:                                                           │
│      flow_duration_mean, iat_mean, iat_std, burstiness              │
│                                                                      │
│    PACKET (aggregated from Tier 2):                                 │
│      ttl_mean, ttl_variance,                                        │
│      tcp_window_mean, tcp_window_std,                               │
│      payload_variance, payload_nonzero_ratio                        │
│                                                                      │
│    SCANNING:                                                         │
│      unique_dst_ports, port_scan_entropy,                           │
│      port_scan_sequential_ratio                                     │
│                                                                      │
│  )  ≈ 60-dimensional vector, RobustScaler-normalised                │
│                                                                      │
│  Sequence for model:  S_{t-9}, S_{t-8}, ..., S_{t-1}, S_t          │
│                       (10 windows = 100 seconds of history)         │
│                                                                      │
│  Optional graph G_t:  nodes = hosts (identified by subnet hash),   │
│                       edges = per-pair behavioural attributes        │
└──────────────────────────────────────────────────────────────────────┘
```

**Window size justification:** 10-second windows provide sufficient temporal resolution to observe attack development (a port scan generating hundreds of probes per second is visible within a single 10s window) while keeping the state vector stable enough for reliable sequence modelling. Window sizes of 5s and 30s are also tested in ablation.

---

### 6.4 Graph-Temporal State Encoder

```
┌──────────────────────────────────────────────────────────────────────┐
│            GRAPH-TEMPORAL STATE ENCODER                             │
│                                                                      │
│  Stage 1 (Optional Advanced): Spatial Encoding                     │
│                                                                      │
│  G_t ──► Graph Attention Network v2 (GAT)                          │
│           • Nodes: host-level subnet hashes (NOT raw IPs)           │
│           • Edges: behavioural attributes per host pair             │
│           • 4 attention heads, 3 message-passing layers             │
│           • Hierarchical pooling: H_t → h_t ∈ R^D                  │
│                                                                      │
│  Note: In the MVP build, this stage is replaced with a simple       │
│  linear projection of the flat S_t vector. GAT is the              │
│  Advanced build extension (see §12).                                │
│                                                                      │
│  Stage 2: Temporal Encoding (both MVP and Advanced)                │
│                                                                      │
│  [S_{t-9}, ..., S_t]  ──►  Temporal Transformer                   │
│   • Positional encoding: time_since_previous_window (real elapsed   │
│     time, NOT just position index)                                  │
│   • Causal masking (no future leakage during training)              │
│   • Output: z_t ∈ R^D — latent network state vector                │
│                                                                      │
│  z_t encodes: "What has this network been doing,                   │
│               in what order, for the last 100 seconds?"            │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 6.5 Temporal World Model Core

```
┌──────────────────────────────────────────────────────────────────────┐
│               STATE TRANSITION NETWORK (STN)                        │
│                                                                      │
│  Input:  z_t  (current latent state)                               │
│                                                                      │
│  Architecture: Probabilistic MLP                                    │
│  z_t ──► [Linear + LayerNorm + ReLU × 2] ──► μ_{t+1}, log σ_{t+1} │
│  Sample: z_{t+1} ~ N(μ_{t+1}, exp(log σ_{t+1}))                   │
│                                                                      │
│  Why probabilistic?                                                 │
│  Network state transitions are inherently stochastic — attacker     │
│  decisions are uncertain, retransmissions are random, and the model │
│  is working from aggregated approximations of truth. A deterministic│
│  predictor gives false precision. The variance σ_{t+1} is a        │
│  meaningful signal: high uncertainty means the trajectory is        │
│  sensitive to small perturbations — which is itself a warning sign. │
│                                                                      │
│  Training objective (joint):                                        │
│  L = α·L_next_state_mse                                            │
│    + β·L_KL (variance regularisation)                              │
│    + γ·L_attack_binary_ce    (attack head)                         │
│    + δ·L_progression_ce      (progression state head)              │
│                                                                      │
│  Training data: CIC-IDS-2017                                       │
│  Ground truth transitions: (S_t, S_{t+1}) pairs from windowed data │
│  Ground truth labels: derived from dataset attack annotations       │
│  (see §8 for label derivation)                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 6.6 Forward Simulation Engine

```
┌──────────────────────────────────────────────────────────────────────┐
│                 K-STEP FORWARD SIMULATION  (K=6)                    │
│                                                                      │
│  Configuration: K=6 steps × 10s = 60 seconds lookahead             │
│  Samples per step: M=50 (Monte Carlo)                               │
│                                                                      │
│  For k = 1 to 6:                                                   │
│    Draw M samples from N(μ_{t+k}, σ_{t+k})                         │
│    For each sample m:                                               │
│      Attack prob:  p_attack^m = Attack Head(z^m_{t+k})             │
│      Progression:  p_state^m  = Progression Head(z^m_{t+k})        │
│                                                                      │
│    Aggregate:                                                        │
│      P(A_{t+k}=1) = mean_m(p_attack^m)                             │
│      P(Z_{t+k}=c) = mean_m(p_state^m[c])  for each state c        │
│      CI_95        = [percentile_2.5, percentile_97.5]              │
│                                                                      │
│  Output per horizon step k:                                         │
│    • P(attack | t+k)  — binary attack probability with CI          │
│    • Z_{t+k}          — most probable progression state            │
│    • P(Z_{t+k}=c)     — full distribution over progression states  │
│    • ATT&CK label     — secondary interpretation (see §8)          │
│                                                                      │
│  Final output: {k: (attack_prob, CI, progression_state,            │
│                     attck_label)} for k in 1..6                    │
│                                                                      │
│  Attack threshold: P(attack) > 0.7  →  HIGH SEVERITY ALERT         │
│  Threshold calibrated to FPR ≤ 5% on validation set                │
└──────────────────────────────────────────────────────────────────────┘
```

**Why K=6 (60 seconds)?** This is grounded in the CIC-IDS-2017 Infiltration scenario: the transition from initial compromise to internal scanning is observable within 1-3 minutes. A 60-second lookahead gives defenders a meaningful action window while keeping rollout error accumulation manageable. K is a configurable hyperparameter; ablation tests K ∈ {3, 6, 12}.

---

### 6.7 Attack Progression State & ATT&CK-Aligned Interpretation

```
┌──────────────────────────────────────────────────────────────────────┐
│          ATTACK PROGRESSION STATES (learned from data)              │
│                                                                      │
│  These five states are the model's PRIMARY learned targets.         │
│  They are derived from CIC-IDS-2017 temporal annotations.           │
│  ATT&CK labels are a SECONDARY semantic layer applied afterward.    │
│                                                                      │
│  NORMAL                                                             │
│    Definition: All flows labelled BENIGN in the dataset window      │
│    ATT&CK: None                                                     │
│                                                                      │
│  PRE-ATTACK / SUSPICIOUS                                            │
│    Definition: Window immediately before confirmed attack onset;    │
│                statistical anomalies not yet matching attack label  │
│    ATT&CK-aligned: Reconnaissance (High confidence for PortScan)   │
│                                                                      │
│  ATTACK-ONSET                                                       │
│    Definition: First window containing a confirmed attack label     │
│    ATT&CK-aligned: Initial Access / Credential Access (Medium)      │
│                                                                      │
│  ACTIVE-ATTACK                                                      │
│    Definition: Sustained attack windows after onset                 │
│    ATT&CK-aligned: Lateral Movement / C2 / Impact (Medium-High,    │
│                    varies by attack type — see §8 table)           │
│                                                                      │
│  POST / CONTINUATION                                                │
│    Definition: Attack label present but behaviour changing          │
│    ATT&CK-aligned: Exfiltration / Impact (Medium)                  │
│                                                                      │
│  ─────────────────────────────────────────────────────────────     │
│  Infiltration-specific sub-states (best demonstration scenario):   │
│                                                                      │
│  NORMAL → EXTERNAL_SUSPICIOUS → INITIAL_COMPROMISE →               │
│  INTERNAL_SCANNING → MULTI_HOST_ACTIVITY                           │
│                                                                      │
│  These are derived from Thursday's Infiltration scenario in         │
│  CIC-IDS-2017, which includes the full progression:                 │
│  external attacker → compromised Vista host → Nmap from victim →   │
│  lateral activity toward other clients.                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 6.8 Explainability Engine

```
┌──────────────────────────────────────────────────────────────────────┐
│                     EXPLAINABILITY ENGINE                           │
│                                                                      │
│  Three complementary mechanisms (applied post-inference):          │
│                                                                      │
│  1. SHAP VALUES (primary explanation for state-level features)     │
│  ─────────────────────────────────────────────────────────────     │
│  • Applied to the attack head output                               │
│  • TreeExplainer for baselines (LR, RF, XGB)                       │
│  • GradientExplainer or KernelExplainer for the Transformer        │
│  • Output: ranked feature contributions per prediction             │
│                                                                      │
│  Example output:                                                    │
│   Attack Probability: 84%  (PRE-ATTACK → ONSET in +30s)           │
│   Top contributing state features:                                 │
│    ↑ port_scan_sequential_ratio  (+0.31 SHAP)                     │
│    ↑ syn_rate                    (+0.27 SHAP)                     │
│    ↑ unique_dst_ports            (+0.19 SHAP)                     │
│    ↓ iat_mean                    (-0.14 SHAP — traffic bursting)  │
│    ↑ failed_connection_ratio     (+0.11 SHAP)                     │
│                                                                      │
│  2. TEMPORAL ATTENTION SALIENCY                                    │
│  ─────────────────────────────────────────────────────────────     │
│  • Extract Transformer attention weights across input windows       │
│  • saliency(t-k) = sum of attention weights pointing to window k   │
│  • Identifies: "the burst 4 windows ago drove this prediction"     │
│  • Displayed as a timeline heatmap in the dashboard                │
│                                                                      │
│  3. GRADIENT-BASED TEMPORAL SALIENCY (Captum)                     │
│  ─────────────────────────────────────────────────────────────     │
│  • Backpropagate prediction score to input state sequence          │
│  • saliency(t-k) = ||∂P(attack)/∂S_{t-k}||₂                      │
│  • More precise than attention alone                               │
│  • Used to validate that attention matches gradient signal         │
│                                                                      │
│  NOTE: Counterfactual attribution (graph-level edge removal) is    │
│  retained as an advanced feature in the GAT build only (§12).     │
│  It is not part of the MVP explainability layer.                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 6.9 Defender Dashboard (Streamlit — fully offline)

```
┌────────────────────────────────────────────────────────────────────────────┐
│  SENTINEL-WM DEFENDER DASHBOARD                               ● LIVE      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  ATTACK PROBABILITY TIMELINE  (60 seconds lookahead, 10s steps)     │ │
│  │                                                                      │ │
│  │  P(att) 1.0│              ╭──────── 95% CI band                     │ │
│  │         0.8│             ╭╯                                          │ │
│  │         0.6│     ─────╮ ╭╯ ← predicted                              │ │
│  │         0.4│ ─────────╰╮╯                                            │ │
│  │         0.2│           ╰─ ← observed                                │ │
│  │         0.0└────────────────────────────────────────── t            │ │
│  │             Now  +10s  +20s  +30s  +40s  +50s  +60s                 │ │
│  │  ⏱ Lead time since first alert:  23 seconds                         │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │  CURRENT STATE: PRE-ATTACK │  │  PREDICTED STATE (+30s): ONSET      │  │
│  │  ██████████ 81%            │  │  ████████░░ 74% (CI: 62–83%)       │  │
│  │                            │  │                                     │  │
│  │  ATT&CK-aligned:           │  │  Progression path:                  │  │
│  │  Reconnaissance (High)     │  │  PRE-ATTACK → ONSET                 │  │
│  │                            │  │                                     │  │
│  │  Top signal features:      │  │  ATT&CK-aligned (+30s):             │  │
│  │  • port_scan_seq_ratio HIGH│  │  Initial Access / Cred. Access (Med)│  │
│  │  • syn_rate ELEVATED       │  │                                     │  │
│  │  • unique_dst_ports: 312   │  │  Recommended actions:               │  │
│  └────────────────────────────┘  │  ① Monitor SSH/FTP brute sources   │  │
│                                  │  ② Alert on new internal pairs     │  │
│                                  └─────────────────────────────────────┘  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  SHAP EXPLANATION  (top 5 state features driving prediction)        │ │
│  │                                                                      │ │
│  │  port_scan_sequential_ratio  ████████████░░░░ +0.31                 │ │
│  │  syn_rate                    ██████████░░░░░░ +0.27                 │ │
│  │  unique_dst_ports            ███████░░░░░░░░░ +0.19                 │ │
│  │  iat_mean                    ░░░░░█████░░░░░░ -0.14                 │ │
│  │  failed_connection_ratio     █████░░░░░░░░░░░ +0.11                 │ │
│  │                                                                      │ │
│  │  Temporal saliency: Windows t-3 and t-5 most salient                │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Strategy

### 7.1 Primary Dataset: CIC-IDS-2017

CIC-IDS-2017 is the primary dataset. It provides both PCAP and labelled CSV flow records, making it uniquely suited for the dual-source architecture.

| Day | CIC-IDS-2017 Attack Scenarios | Use |
|---|---|---|
| Monday | BENIGN only | Baseline benign training |
| Tuesday | FTP-Patator, SSH-Patator | Credential Access training |
| Wednesday | DoS Slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed | DoS/Exploit training |
| Thursday | Web Attack – Brute Force, Web Attack – XSS, Web Attack – SQL Injection, **Infiltration** | Multi-stage training; **primary demo scenario** |
| Friday | Botnet ARES, PortScan, DDoS LOIC | C2, Recon, Impact training |

> **Note:** The original proposal incorrectly cited CIC-IDS-2018. The primary dataset is **CIC-IDS-2017**, which matches both the attack diversity and the PCAP+CSV dual availability required by this architecture.

### 7.2 Secondary Dataset: CTU-13

CTU-13 NetFlow records are used for out-of-distribution evaluation — testing whether the trained model generalises to botnet and lateral movement patterns from a different network environment.

| Dataset | Type | Use |
|---|---|---|
| CIC-IDS-2017 | PCAP + CSV | Primary training and evaluation |
| CTU-13 | NetFlow | Cross-dataset generalisation test |
| UNSW-NB15 | CSV | Supplementary evaluation on unseen attack families |

### 7.3 State Window Construction

```
Data Preprocessing Pipeline:

1. Load CIC-IDS-2017 CSV (per-day files), sort by Timestamp
2. Parse corresponding PCAP files, extract per-session Tier 2 features
3. Join on 5-tuple + flow start timestamp (δ = ±2 seconds tolerance)
4. Construct 10-second sliding windows (stride = 5 seconds, 50% overlap)
5. Per window: aggregate into S_t (Tier 1 + 2 + 3 features, ~60 dims)
6. Derive progression state label from annotations:
     if window is all BENIGN:                    state = NORMAL
     if window precedes first attack by 1-3w:   state = PRE-ATTACK
     if window contains first attack label:      state = ONSET
     if window is sustained attack:              state = ACTIVE
     if attack label changes / dissipates:       state = CONTINUATION
7. Derive binary attack label:
     A_t = 0 if NORMAL, 1 otherwise
8. Build sequence dataset: [(S_{t-9:t}, A_t, Z_t)] for all t
9. Build transition pairs: (S_{t-9:t}, S_{t+1}) for STN training

⚠️  CRITICAL — Train/Validation/Test Split by DAY, never by flow:
     TRAIN:       Monday + Tuesday + Wednesday
     VALIDATION:  Thursday
     TEST:        Friday
     
     Rationale: Random-row splitting randomises time, causing temporal
     leakage. A model that has seen "Friday flows" during training can
     trivially predict "Friday attacks" — this does not prove forecasting
     capability, it proves memorisation. Day-based splitting ensures the
     test set is genuinely unseen attack timing.
```

### 7.4 Class Imbalance Handling

CIC-IDS-2017 is heavily imbalanced (Monday is entirely benign; attack flows are 10-30% of total). Mitigations:

- **Weighted cross-entropy loss** with class weights inversely proportional to frequency
- **Stratified window sampling** ensuring each training batch contains a minimum 30% attack windows
- **Separate threshold calibration**: decision threshold for P(attack) > θ is calibrated on validation set to constrain FPR ≤ 5%

### 7.5 Data Augmentation for Generalisation

- **Port shuffle augmentation:** Randomly remap destination ports within a window to teach behavioural pattern learning rather than port memorisation
- **Timing jitter:** ±15% noise on IAT values to simulate attacker timing obfuscation
- **Window dropout:** Randomly drop 20% of flows within a training window to simulate TAP-based collection gaps
- **Slow-scan synthesis:** Split high-rate scans into synthetically slowed sub-sequences to expose model to low-and-slow variants

---

## 8. MITRE ATT&CK Integration — Corrected Approach

### 8.1 What CIC-IDS-2017 Can and Cannot Support

> **Important constraint acknowledged:** CIC-IDS-2017 provides **attack type labels per flow**, not **MITRE ATT&CK tactic labels per time window**. The dataset tells you "this flow is PortScan" — it does not tell you "at this moment, the attacker is in the Reconnaissance tactic."
>
> Therefore, MITRE ATT&CK phase labels are **secondary semantic interpretations** of the model's learned progression states, not primary supervised targets. This distinction is made explicit in all outputs and documentation.

### 8.2 CIC-IDS-2017 Attack Label → ATT&CK-Aligned Interpretation Table

| CIC-IDS-2017 Label | ATT&CK-Aligned Interpretation | Confidence |
|---|---|---|
| BENIGN | None | High |
| PortScan | Reconnaissance / Discovery (T1595, T1046) | High |
| FTP-Patator | Credential Access / Initial Access (T1110, T1190) | Medium |
| SSH-Patator | Credential Access / Initial Access (T1110, T1021) | Medium |
| Web Attack – Brute Force | Credential Access (T1110.004) | Medium |
| Web Attack – SQL Injection | Initial Access / Execution (T1190, T1059) | Medium |
| Web Attack – XSS | Initial Access (T1190) | Medium |
| Heartbleed | Initial Access / Credential Access (T1190) | Medium |
| **Infiltration** | **Initial Access → Discovery → Lateral Movement (multi-phase)** | **Medium-High** |
| Botnet ARES | Command & Control (T1071, T1572) | Medium |
| DoS Slowloris | Impact (T1499) | High |
| DoS Slowhttptest | Impact (T1499) | High |
| DoS Hulk | Impact (T1499) | High |
| DoS GoldenEye | Impact (T1499) | High |
| DDoS LOIC | Impact (T1498) | High |

**All ATT&CK mappings are labelled with their confidence level in all system outputs.** The system never presents these as ground-truth phase annotations.

### 8.3 The Infiltration Scenario — Primary Demonstration Case

Thursday's Infiltration scenario in CIC-IDS-2017 is the most compelling test case because it contains a genuine multi-phase progression:

```
External attacker activity
        ↓
Compromise of internal Windows Vista host
        ↓
Nmap scanning FROM the victim host toward other internal clients
        ↓
Multi-host activity

SENTINEL-WM learns to predict this trajectory:
S_NORMAL → S_EXTERNAL_SUSPICIOUS → S_INITIAL_COMPROMISE →
S_INTERNAL_SCANNING → S_MULTI_HOST_ACTIVITY

Forward simulation output at S_EXTERNAL_SUSPICIOUS:
  P(A_{t+3}=1) = 0.82  (30 seconds ahead)
  Z_{t+3} = INTERNAL_SCANNING  (most probable state at +30s)
  ATT&CK-aligned: Discovery (T1046) [Medium]

This is the clearest demonstration of genuine temporal forecasting capability
in the dataset — the model warns of internal scanning before it begins,
based solely on external traffic patterns.
```

---

## 9. System Workflows

### 9.1 Training Workflow

```
╔══════════════════════════════════════════════════════════════════════╗
║                        TRAINING WORKFLOW                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. DATA PREPARATION                                              ║
║     ├── Download CIC-IDS-2017 (per-day CSVs + PCAPs)             ║
║     ├── Parse PCAP → Tier 2 features (Scapy, parallel)           ║
║     ├── Join CSV flows + PCAP sessions on 5-tuple + timestamp     ║
║     └── Assign split: Mon-Wed=Train, Thu=Val, Fri=Test           ║
║                         │                                         ║
║  2. STATE WINDOW CONSTRUCTION                                     ║
║     ├── 10-second windows, 5-second stride                        ║
║     ├── Compute Tier 3 behavioural features per window           ║
║     ├── Normalise (RobustScaler fitted on TRAIN only)            ║
║     └── Assign progression state and binary attack labels        ║
║                         │                                         ║
║  3. PHASE 1: BASELINE TRAINING                                   ║
║     ├── Train Logistic Regression on S_t → A_t                   ║
║     ├── Train Random Forest / XGBoost on S_t → A_t               ║
║     └── Record F1, Precision, Recall, FPR, Lead Time             ║
║                         │                                         ║
║  4. PHASE 2: TEMPORAL WORLD MODEL TRAINING (MVP)                 ║
║     ├── Input: sequences [S_{t-9:t}]                             ║
║     ├── Temporal Transformer → z_t                               ║
║     ├── Attack Head: z_t → P(A_t)                                ║
║     ├── Progression Head: z_t → P(Z_t=c)                         ║
║     ├── STN: z_t → μ_{t+1}, σ_{t+1}                             ║
║     └── Joint loss optimisation (Adam, LR=1e-4)                  ║
║                         │                                         ║
║  5. PHASE 3: CALIBRATION                                         ║
║     ├── Compute ECE and Brier Score on validation set            ║
║     ├── Calibrate decision threshold to FPR ≤ 5%                ║
║     └── Reliability diagram generation                           ║
║                         │                                         ║
║  6. PHASE 4: EXPLAINABILITY EXTRACTION                           ║
║     ├── Compute SHAP values on validation predictions            ║
║     ├── Validate temporal attention saliency                     ║
║     └── Cross-check with known attack indicators                 ║
║                         │                                         ║
║  7. CHECKPOINT & EXPORT                                          ║
║     ├── Save weights (.pt), scaler, config (YAML)                ║
║     ├── Save per-attack lead time analysis                        ║
║     └── Generate training report (metrics + ablation table)     ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 9.2 Inference Workflow (Batch: PCAP or CSV input)

```
╔══════════════════════════════════════════════════════════════════════╗
║                       INFERENCE WORKFLOW                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  INPUT: PCAP file  OR  CSV file (flow records)                    ║
║               │                                                    ║
║   ┌───────────▼────────────────────────────────────────────────┐  ║
║   │  Feature extraction (Tier 1 from CSV, Tier 2 from PCAP)   │  ║
║   │  If only CSV provided: Tier 2 defaults to 0               │  ║
║   └───────────┬────────────────────────────────────────────────┘  ║
║               │                                                    ║
║   ┌───────────▼────────────────────────────────────────────────┐  ║
║   │  10-second window construction → S_t sequence             │  ║
║   │  RobustScaler normalisation (using training scaler)       │  ║
║   └───────────┬────────────────────────────────────────────────┘  ║
║               │                                                    ║
║   ┌───────────▼────────────────────────────────────────────────┐  ║
║   │  Temporal Transformer: [S_{t-9:t}] → z_t                 │  ║
║   └───────────┬────────────────────────────────────────────────┘  ║
║               │                                                    ║
║   ┌───────────▼────────────────────────────────────────────────┐  ║
║   │  K=6 step MC rollout → {P(A_{t+k}), Z_{t+k}, CI}         │  ║
║   └───────────┬────────────────────────────────────────────────┘  ║
║               │                                                    ║
║   ┌───────────▼────────────────────────────────────────────────┐  ║
║   │  SHAP + attention extraction → top contributing features  │  ║
║   └───────────┬────────────────────────────────────────────────┘  ║
║               │                                                    ║
║  OUTPUT FILES:                                                    ║
║   ├── attack_probability_timeline.json                           ║
║   ├── progression_state_predictions.json                         ║
║   ├── attck_aligned_interpretation.json                          ║
║   ├── shap_explanations.json                                     ║
║   └── streamlit dashboard (auto-launched)                        ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 9.3 Defender Action Loop

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DEFENDER ACTION LOOP                            │
│                                                                      │
│  SENTINEL-WM output per 10-second cycle:                           │
│                │                                                    │
│  ┌─────────────▼──────────────────────────────────────────────┐    │
│  │  P(attack | t+k) > 0.7 for any k ≤ 6?                    │    │
│  │  YES ──► HIGH SEVERITY ALERT                               │    │
│  │           + Lead time: (k × 10s) before predicted onset   │    │
│  │           + Progression state + ATT&CK-aligned label       │    │
│  │           + Top 5 SHAP features driving the prediction     │    │
│  │  NO  ──► Continue monitoring, update rolling window        │    │
│  └─────────────┬──────────────────────────────────────────────┘    │
│                │                                                    │
│  ┌─────────────▼──────────────────────────────────────────────┐    │
│  │  Defender response (manual or automated):                  │    │
│  │  • Block egress from hosts with high port_scan_seq_ratio   │    │
│  │  • Isolate host pairs showing new_connections spike        │    │
│  │  • Trigger EDR scan on fan_out outliers                    │    │
│  │  • Invoke IR playbook matched to predicted ATT&CK label    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. Evaluation & Benchmarking Strategy

### 10.1 Metrics — Complete Set

| Category | Metric | Formula | Why It Matters |
|---|---|---|---|
| Classification | F1 Score (per state) | 2·P·R / (P+R) | Stage prediction quality |
| Classification | Precision | TP / (TP+FP) | False alert cost |
| Classification | Recall | TP / (TP+FN) | Missed attack cost |
| Classification | False Positive Rate | FP / (FP+TN) | Defender fatigue |
| Classification | AUROC | Area under ROC curve | Overall discrimination |
| **Forecasting** | **Mean Lead Time (MLT)** | **avg(t_alert − t_attack_start)** | **Core innovation metric** |
| Forecasting | Time-to-Warning (TTW) | t_attack_start − t_first_warning | Operational headroom |
| Forecasting | False Alarm Rate | false warnings / benign windows | Operational trust |
| Forecasting | Horizon Accuracy | F1 per k ∈ {+10s, +20s, +30s, +60s} | Forecasting reliability vs horizon |
| Calibration | Brier Score | mean((P_pred − P_true)²) | Probability reliability |
| Calibration | ECE | Expected calibration error | Confidence interval trust |
| Calibration | Reliability Diagram | Binned accuracy vs predicted prob | Visual calibration check |

**Mean Lead Time** and **Forecast Horizon Accuracy** are the primary differentiating metrics that prove temporal forecasting beyond static classification.

### 10.2 Baseline Comparisons

| Model | Architecture | Input | Limitation Demonstrated |
|---|---|---|---|
| Baseline 1: Logistic Regression | Linear classifier | Current window S_t | No temporal modelling |
| Baseline 2: Random Forest / XGBoost | Tree ensemble | Current window S_t | No temporal modelling |
| Baseline 3: LSTM Classifier | Sequence → binary label | S_{t-9:t} | No forward simulation, no stage |
| **SENTINEL-WM MVP** | Temporal Transformer + STN | S_{t-9:t} | — |
| **SENTINEL-WM Advanced** | GAT + Temporal Transformer + STN | G_{t-9:t} + S_{t-9:t} | — |

### 10.3 Forecast Horizon Accuracy Table (Target)

| Horizon | Target F1 | Target AUROC | Expected degradation |
|---|---|---|---|
| +10s (k=1) | > 0.88 | > 0.93 | Minimal |
| +20s (k=2) | > 0.85 | > 0.91 | Low |
| +30s (k=3) | > 0.82 | > 0.89 | Moderate |
| +60s (k=6) | > 0.76 | > 0.85 | Expected; CI widens |

### 10.4 Lead Time Target (Infiltration Scenario)

| Detection point | Target MLT | Notes |
|---|---|---|
| PRE-ATTACK → ONSET | > 30 seconds | Warning before first attack label |
| ONSET → INTERNAL_SCANNING | > 40 seconds | Warning before lateral phase begins |
| Median TTW across all attack days | > 25 seconds | Across all attack families |

### 10.5 Ablation Study Design

| Ablation | What is removed | Expected impact |
|---|---|---|
| No Tier 2 (packet) features | Packet features zeroed | Drop in Recon detection (TTL, scan scores) |
| No temporal history | Single window S_t only → degenerates to Baseline 1 | Large drop in MLT |
| No uncertainty (deterministic STN) | σ removed, point prediction only | No CI output; similar point F1 |
| 5s vs 10s vs 30s window | Window size varies | Resolution vs stability trade-off |
| K=3 vs K=6 vs K=12 | Rollout horizon varies | Accuracy vs lead time trade-off |

---

## 11. Key Innovations & Justifications

### 11.1 Innovation: Dual-Source, Tiered Feature Architecture with Explicit Authority

**Justification:** Mixing packet and flow features without attribution of source is a common architecture smell that creates untestable pipelines. By designating the CSV as authoritative for flow features and the PCAP as authoritative for packet features — joined deterministically on 5-tuple + timestamp — the system is fully reproducible, auditable, and debuggable. Any reviewer can independently verify which features come from which source.

### 11.2 Innovation: Network State S_t as the Primary Object

**Justification:** A 10-second window aggregation separates the model from the noise of individual flow variation. A single TCP connection can look anomalous for benign reasons (retransmission, slow link). But 312 unique destination ports from one source in 10 seconds is unambiguously scanning behaviour, regardless of per-flow variability. The state vector captures what matters for attack progression — aggregate patterns — not per-flow statistics that carry too much noise for sequence modelling.

### 11.3 Innovation: Probabilistic State Transition (Uncertainty as a Feature)

**Justification:** High uncertainty on the STN output (large σ) is itself operationally meaningful — it signals that the network is in a state with many possible futures, which often corresponds to the moment just before an attacker makes a branching decision (e.g., which host to pivot to). Presenting uncertainty as a calibrated CI empowers defenders to act on the worst-case tail — a decision framework that static classifiers cannot support.

### 11.4 Innovation: ATT&CK as Interpretation Layer, Not Ground Truth

**Justification:** Forcing ATT&CK phase labels as primary supervised targets on CIC-IDS-2017 is methodologically unsound — the dataset's attack annotations do not carry ground-truth tactic-level granularity. Systems that do this are making an epistemically dishonest claim that reviewers familiar with ATT&CK will challenge. SENTINEL-WM uses data-driven progression states as the primary learned target, then applies ATT&CK-aligned labels as a secondary semantic layer with explicit confidence scores. This is both more scientifically honest and more robust to expert scrutiny.

### 11.5 Innovation: Lead Time as the Primary Operational Metric

**Justification:** F1 score tells you how accurate the model was *at the moment of prediction*. Lead time tells you how much *time the defender had to act*. These are categorically different. A system with F1=0.92 and 0-second lead time is operationally useless. A system with F1=0.83 and 40-second lead time is genuinely valuable. SENTINEL-WM treats lead time as the primary success criterion because that is the metric that maps directly to operational value in a SOC or CIRT context.

### 11.6 Innovation: Day-Based Train/Val/Test Split Preventing Temporal Leakage

**Justification:** Random-row splitting on CIC-IDS-2017 produces artificially inflated metrics because the model trains on flows from the same attack sessions it is then tested on. Day-based splitting — training on Mon-Wed, validating on Thursday, testing on Friday — ensures the model is evaluated on genuinely unseen attack timing and, in the case of the Botnet and DDoS scenarios on Friday, partially unseen attack families. This is the only split strategy that demonstrates real generalisation.

---

## 12. Three-Tier Build Strategy

> **Rationale:** A proposal that promises GAT + Variational SSM + MC rollout + SHAP + counterfactuals + live traffic simultaneously risks spending the entire project debugging architecture rather than proving forecasting capability. The three-tier strategy ensures a working, evaluable system exists at every stage.

### Tier 1 — Baseline (Week 1-2, immediate deliverable)

```
Purpose:  Establish comparison floor. Prove the data pipeline works.

Models:   Logistic Regression + Random Forest / XGBoost
Input:    S_t (current window only)
Output:   Binary attack label
Metrics:  F1, Precision, Recall, FPR, AUROC

Deliverable: Working CLI that ingests CSV, constructs windows,
             trains baselines, and reports metrics.
```

### Tier 2 — MVP World Model (Week 3-6, core deliverable)

```
Purpose:  Prove temporal forecasting advantage over Tier 1.

Model:    Temporal Transformer + probabilistic STN
Input:    S_{t-9:t} (10-window sequence, flat state vectors)
Output:   P(A_{t+k}) for k=1..6, progression state Z_{t+k},
          SHAP explanations, confidence intervals
Metrics:  All Tier 1 metrics + MLT + TTW + Horizon Accuracy
          + Brier Score + ECE

Deliverable: Streamlit dashboard, trained model weights,
             evaluation report with baseline comparison.
```

### Tier 3 — Advanced Extension (Week 7-10, if time permits)

```
Purpose:  Demonstrate graph-topology-aware state encoding.

Model:    GAT spatial encoder + Temporal Transformer + STN
Input:    G_{t-9:t} (graph sequence) + S_{t-9:t} (vector sequence)
Output:   All MVP outputs + graph attention visualisation
          + counterfactual edge attribution
Metrics:  All MVP metrics + ablation against MVP baseline

Deliverable: Extended system with graph visualisation panel.
             Documented comparison: GAT vs flat-vector encoding.
```

---

## 13. Technology Stack

### Core Components

| Component | Technology | Rationale |
|---|---|---|
| Temporal Transformer | PyTorch + custom positional encoding | Fine-grained control over causal masking; real elapsed-time positional encoding |
| Probabilistic STN | PyTorch + `torch.distributions.Normal` | Clean API for Gaussian state transition |
| GAT (Advanced) | PyTorch Geometric (PyG) — GAT v2 | Industry-standard; well-documented; GAT v2 avoids static attention problem |
| PCAP Parsing | Scapy 2.x | Battle-tested; handles malformed packets; session aggregation scriptable |
| Flow Features | CIC-IDS-2017 CSV directly (pandas/polars) | polars for speed on >70GB day files |
| SHAP | SHAP library (GradientExplainer) | Consistent API across baselines and deep model |
| Temporal Saliency | Captum (PyTorch) | Gradient-based attribution; integrates with PyTorch autograd |
| Dashboard | Streamlit | Rapid prototyping; fully offline on port 8501 |
| Visualisation | Plotly + pyvis (graph view in Advanced tier) | Interactive timeline + network graph rendering |
| Baselines | scikit-learn | LR, RF; metric computation |
| Experiment Tracking | MLflow (local, no cloud) | Reproducible training configs; run comparison |
| Normalisation | scikit-learn RobustScaler | Handles outliers from traffic bursts better than StandardScaler |

### No Cloud Dependencies

All components run fully offline. The PCAP and CSV files are stored locally. The Streamlit dashboard runs on localhost:8501. MLflow tracking is file-based. No external API calls are made at any stage.

---

## 14. Deployment Architecture

### 14.1 Hardware Requirements

| Environment | Minimum | Recommended |
|---|---|---|
| PCAP feature extraction | 16GB RAM, 8 CPU cores, 1TB SSD | 32GB RAM, 16 cores, 2TB NVMe |
| Model training | 16GB RAM, NVIDIA RTX 3060, 500GB | 32GB RAM, NVIDIA RTX 4080, 1TB NVMe |
| Inference (batch) | 8GB RAM, CPU only (slower) | 16GB RAM, any CUDA GPU |
| Dashboard | Any modern browser | — |

### 14.2 Offline Deployment Package Structure

```
sentinel-wm/
├── README.md
├── requirements.txt
├── configs/
│   ├── model_config.yaml          # Architecture hyperparameters
│   ├── feature_config.yaml        # Tier 1/2/3 column specs, window size
│   └── evaluation_config.yaml     # Split dates, threshold, K horizon
├── data/
│   └── README_data_download.md    # CIC-IDS-2017 download instructions
├── src/
│   ├── ingestion/
│   │   ├── pcap_extractor.py      # Tier 2: Scapy-based packet features
│   │   ├── flow_loader.py         # Tier 1: CIC-IDS-2017 CSV loader
│   │   ├── flow_joiner.py         # 5-tuple + timestamp join logic
│   │   └── state_builder.py       # Tier 3 + Network State S_t constructor
│   ├── models/
│   │   ├── temporal_transformer.py # Temporal state encoder
│   │   ├── state_transition.py    # Probabilistic STN (μ, σ output)
│   │   ├── attack_head.py         # Binary attack probability head
│   │   ├── progression_head.py    # Progression state classifier
│   │   └── gat_encoder.py         # [Advanced] GAT spatial encoder
│   ├── simulation/
│   │   └── forward_sim.py         # K-step MC rollout engine
│   ├── explainability/
│   │   ├── shap_explainer.py      # SHAP extraction for attack head
│   │   ├── attention_saliency.py  # Transformer attention visualisation
│   │   └── temporal_gradient.py   # Captum gradient-based saliency
│   ├── evaluation/
│   │   ├── metrics.py             # F1, FPR, AUROC, Brier, ECE
│   │   ├── lead_time.py           # MLT and TTW computation
│   │   └── horizon_eval.py        # Forecast horizon accuracy table
│   ├── training/
│   │   ├── train.py               # Main training script (joint loss)
│   │   ├── dataset.py             # PyTorch Dataset + DataLoader
│   │   └── losses.py              # L_mse + L_KL + L_attack + L_progression
│   └── dashboard/
│       └── app.py                 # Streamlit dashboard (fully offline)
├── scripts/
│   ├── extract_pcap_features.sh   # Parallel Scapy extraction
│   ├── build_windows.sh           # State window construction
│   ├── train_baselines.sh
│   ├── train_worldmodel.sh
│   └── run_inference.sh           # Batch PCAP or CSV inference
├── weights/
│   └── sentinel_wm_v1.pt          # Trained model weights
└── tests/
    ├── test_flow_loader.py
    ├── test_pcap_extractor.py
    ├── test_state_builder.py
    ├── test_forward_sim.py
    └── test_lead_time.py
```

---

## 15. Risk Analysis & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Temporal data leakage (random split) | **Critical** | Day-based splits hardcoded in config; CI tests verify no test timestamps appear in train set |
| IP address memorisation | **High** | Raw IPs never fed to model; only subnet hash, is_internal, host pair novelty flags used |
| source_file leakage | **High** | source_file column explicitly excluded from all model inputs; used only for split assignment |
| raw timestamp as feature | **High** | Timestamp used only for ordering; model input uses time_since_previous_window only |
| MITRE ATT&CK labels presented as ground truth | **High** | All ATT&CK labels marked with confidence level; primary targets are data-driven progression states |
| MC rollout error accumulation at large K | Medium | K capped at 12; CI widens automatically with rollout depth; dashboard shows CI width as a reliability signal |
| High false positive rate on benign windows | Medium | Decision threshold calibrated to FPR ≤ 5% constraint on validation; separate benign-day baseline characterised |
| PCAP parsing performance on 70GB dataset | Medium | Parallel Scapy extraction (multiprocessing pool); feature cache stored to disk after first run |
| Overfitting to CIC-IDS-2017 topology | Medium | Port shuffle augmentation; cross-evaluation on CTU-13; entity anonymisation via subnet hashing |
| Claiming GAT is validated when only MVP runs | Low | Three-tier strategy makes MVP and Advanced explicitly distinct deliverables; GAT results clearly labelled as Advanced |

---

## 16. Appendix: Feature Reference Tables

### A1. Complete Locked Feature Schema (All Tiers)

#### Tier 1 — CICFlowMeter-Derived Flow Features (44 columns)

| Group | # | Feature | Source |
|---|---|---|---|
| Identity* | 1 | Src IP → `src_host_id` | CSV → derived |
| Identity* | 2 | Src Port | CSV |
| Identity* | 3 | Dst IP → `dst_host_id` | CSV → derived |
| Identity* | 4 | Dst Port | CSV |
| Identity* | 5 | Protocol | CSV |
| Identity* | 6 | Timestamp | CSV → ordering only |
| Volume | 7 | Flow Duration | CSV |
| Volume | 8 | Total Fwd Packets | CSV |
| Volume | 9 | Total Backward Packets | CSV |
| Volume | 10 | Total Length of Fwd Packets | CSV |
| Volume | 11 | Total Length of Bwd Packets | CSV |
| Pkt Length | 12-15 | Fwd Packet Length Max/Min/Mean/Std | CSV |
| Pkt Length | 16-19 | Bwd Packet Length Max/Min/Mean/Std | CSV |
| Pkt Length | 20-22 | Min/Max Packet Length, Packet Length Mean | CSV |
| Pkt Length | 23-24 | Packet Length Std, Packet Length Variance | CSV |
| Rate | 25 | Flow Bytes/s | CSV |
| Rate | 26 | Flow Packets/s | CSV |
| Rate | 27 | Fwd Packets/s | CSV |
| Rate | 28 | Bwd Packets/s | CSV |
| IAT | 29-33 | Flow IAT Mean/Std/Max/Min/Variance | CSV |
| IAT | 34-38 | Fwd IAT Total/Mean/Std/Max/Min | CSV |
| IAT | 39-43 | Bwd IAT Total/Mean/Std/Max/Min | CSV |
| TCP Flags | 44-45 | Fwd PSH Flags, Bwd PSH Flags | CSV |
| TCP Flags | 46-47 | Fwd URG Flags, Bwd URG Flags | CSV |
| TCP Flags | 48-55 | FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Count | CSV |
| Header/Win | 56-57 | Fwd Header Length, Bwd Header Length | CSV |
| Header/Win | 58-59 | Init_Win_bytes_forward, Init_Win_bytes_backward | CSV |
| Header/Win | 60 | min_seg_size_forward | CSV |
| Header/Win | 61 | act_data_pkt_fwd | CSV |
| Ratio/Active | 62 | Down/Up Ratio | CSV |
| Ratio/Active | 63-66 | Active Mean/Std/Max/Min | CSV |
| Ratio/Active | 67-70 | Idle Mean/Std/Max/Min | CSV |

> *Identity features used for graph construction and flow joining only; raw IP values never fed into neural network layers.

#### Tier 2 — Packet Features (18 locked columns, PCAP-derived)

| # | Feature | Source | Level |
|---|---|---|---|
| 1 | ttl_mean | PCAP / Scapy | Packet |
| 2 | ttl_variance | PCAP / Scapy | Packet |
| 3 | tcp_window_mean | PCAP / Scapy | Packet |
| 4 | tcp_window_std | PCAP / Scapy | Packet |
| 5 | frag_more_flag_present | PCAP / Scapy | Packet |
| 6 | frag_dont_flag_present | PCAP / Scapy | Packet |
| 7 | frag_max_offset | PCAP / Scapy | Packet |
| 8 | payload_min | PCAP / Scapy | Packet |
| 9 | payload_max | PCAP / Scapy | Packet |
| 10 | payload_variance | PCAP / Scapy | Packet |
| 11 | payload_skew | PCAP / Scapy | Packet |
| 12 | payload_kurtosis | PCAP / Scapy | Packet |
| 13 | payload_nonzero_ratio | PCAP / Scapy | Packet |
| 14 | port_scan_entropy | PCAP / Scapy | Behaviour |
| 15 | unique_dst_ports_per_src | PCAP / Scapy | Behaviour |
| 16 | port_scan_max_sequential_run | PCAP / Scapy | Behaviour |
| 17 | port_scan_sequential_ratio | PCAP / Scapy | Behaviour |
| 18 | retransmission_count | PCAP / Scapy | Packet |

#### Tier 4 — Metadata (excluded from all model inputs)

| Field | Use | Why excluded |
|---|---|---|
| `timestamp_window` | Sequence ordering, evaluation, visualisation | Raw timestamp allows day-of-week memorisation |
| `source_file` | Data lineage, split assignment, debugging | Directly encodes train/test split identity |

### A2. CIC-IDS-2017 Attack Day Schedule (Corrected)

| Day | Date | Attack Scenarios |
|---|---|---|
| Monday | 2017-07-03 | BENIGN |
| Tuesday | 2017-07-04 | FTP-Patator, SSH-Patator |
| Wednesday | 2017-07-05 | DoS Slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed |
| Thursday | 2017-07-06 | Web Attack – Brute Force, Web Attack – XSS, Web Attack – SQL Injection, **Infiltration** |
| Friday | 2017-07-07 | **Botnet ARES**, **PortScan**, **DDoS LOIC** |

### A3. CIC-IDS-2017 Label → Progression State → ATT&CK Interpretation

| CIC-IDS-2017 Label | Progression State | ATT&CK-Aligned Interpretation | Confidence |
|---|---|---|---|
| BENIGN | NORMAL | None | High |
| PortScan | PRE-ATTACK / ONSET | Reconnaissance / Discovery (T1595, T1046) | High |
| FTP-Patator | ONSET / ACTIVE | Credential Access / Initial Access (T1110) | Medium |
| SSH-Patator | ONSET / ACTIVE | Credential Access / Initial Access (T1110) | Medium |
| DoS Slowloris | ACTIVE | Impact (T1499) | High |
| DoS Slowhttptest | ACTIVE | Impact (T1499) | High |
| DoS Hulk | ACTIVE | Impact (T1499) | High |
| DoS GoldenEye | ACTIVE | Impact (T1499) | High |
| Heartbleed | ONSET | Initial Access / Credential Exposure (T1190) | Medium |
| Web Attack – Brute Force | ONSET / ACTIVE | Credential Access (T1110.004) | Medium |
| Web Attack – XSS | ONSET / ACTIVE | Initial Access (T1190) | Medium |
| Web Attack – SQL Injection | ONSET / ACTIVE | Initial Access / Execution (T1190, T1059) | Medium |
| **Infiltration** | **All states** | **Multi-phase: Recon → Initial Access → Discovery → Lateral Movement** | **Medium-High** |
| Botnet ARES | ACTIVE / CONTINUATION | Command & Control (T1071, T1572) | Medium |
| DDoS LOIC | ACTIVE | Impact (T1498) | High |

### A4. Review Validation Scorecard

| Item Reviewed | Original Proposal | Corrected in v2.0 | Status |
|---|---|---|---|
| Primary dataset | CIC-IDS-2018 ❌ | CIC-IDS-2017 ✓ | Fixed |
| Feature description | "CICFlowMeter 44 columns" ❌ | "CICFlowMeter-derived subset" ✓ | Fixed |
| Raw IP as model feature | Yes ❌ | No; subnet hash + behavioural derivatives ✓ | Fixed |
| source_file as model feature | Implicit ❌ | Explicitly excluded (Tier 4) ✓ | Fixed |
| Timestamp as model feature | Implicit ❌ | Ordering only; time_since_prev used ✓ | Fixed |
| Train/test split | 70/15/15 random rows ❌ | Day-based: Mon-Wed / Thu / Fri ✓ | Fixed |
| MITRE as primary ground truth | Yes ❌ | Secondary interpretation layer ✓ | Fixed |
| "MITRE attractors" terminology | Yes (risky) | Replaced: ATT&CK-aligned interpretation ✓ | Fixed |
| Window size | 60s | 10s (recommended); 5s/30s ablated ✓ | Updated |
| Evaluation metrics | F1, Precision, Recall, FPR, Brier, ECE | + Lead Time, TTW, Horizon Accuracy ✓ | Extended |
| Infiltration as primary demo | Not highlighted | Elevated as primary scenario ✓ | Updated |
| Build complexity | All-at-once ❌ | Three-tier strategy ✓ | Added |
| Packet feature source | Mixed (some from CSV) ❌ | PCAP exclusively (Tier 2) ✓ | Fixed |
| Feature tier separation | Implicit | Explicit: Flow / Packet / Behaviour / Metadata ✓ | Added |

---

*SENTINEL-WM is proposed as a fully open-source research prototype.*
*All datasets used are publicly available for academic and research purposes.*
*No proprietary data, APIs, or cloud dependencies are required at any stage.*

---

**Document Version:** 2.0 — Post-Review Update
**Review basis:** CIC-IDS-2017 dataset structure verification, MITRE ATT&CK framework alignment, feature feasibility analysis, temporal leakage prevention audit
**Classification:** Unclassified / For Evaluation
ENDOFFILE