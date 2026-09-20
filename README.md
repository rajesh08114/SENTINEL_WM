# SENTINEL-WM

**S**patio-Temporal **E**nemy **N**etwork **I**ntelligence with a Learned **World Model**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Next.js: 14](https://img.shields.io/badge/Next.js-14%20App%20Router-black.svg)](https://nextjs.org/)
[![FastAPI: 0.110+](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Packet Inspection: Npcap / Wireshark](https://img.shields.io/badge/Capture-Npcap%20%7C%20Wireshark%20BPF-1679A7.svg)](https://npcap.com/)

> **Proactive Cyber Defence via Temporal World Models**  
> Rather than classifying isolated network flows as benign or malicious in retrospect, SENTINEL-WM aggregates network traffic into **10-second dynamical state windows**, learns temporal state-transition mechanics ($P(S_{t+1} \mid S_{t-L:t})$), and **rolls the latent world forward $K = 6$ steps (10 to 60 seconds into the future)** via Monte Carlo simulations.  
>
> The system predicts attacker progression stages, maps prospective MITRE ATT&CK tactics, evaluates deterministic security rules, and produces driving-feature explanations **before the adversary's kill chain completes**—granting defenders actionable lead time to intervene proactively.

---

## Table of Contents

1. [Problem Context — The Defender's Dilemma](#problem-context--the-defenders-dilemma)
2. [Key Advantages & Capabilities](#key-advantages--capabilities)
3. [End-to-End System Architecture](#end-to-end-system-architecture)
4. [Prerequisites & System Requirements](#prerequisites--system-requirements)
5. [Installation & Setup Guide](#installation--setup-guide)
   - [Option A: Local Development Setup](#option-a-local-development-setup)
   - [Option B: Containerized Deployment (Docker Compose)](#option-b-containerized-deployment-docker-compose)
6. [How to Use the Platform](#how-to-use-the-platform)
   - [1. Instant Demo Flow Evaluation](#1-instant-demo-flow-evaluation)
   - [2. Wireshark PCAP / PCAPNG Ingestion with BPF Filtering](#2-wireshark-pcap--pcapng-ingestion-with-bpf-filtering)
   - [3. Live Network Interface Sniffing (Host Capture Agent)](#3-live-network-interface-sniffing-host-capture-agent)
   - [4. Custom Security & Heuristic Rules Engine](#4-custom-security--heuristic-rules-engine)
   - [5. SOC Analyst Console Walkthrough](#5-soc-analyst-console-walkthrough)
7. [Inference Engine & API Reference](#inference-engine--api-reference)
8. [Empirical Benchmark Results](#empirical-benchmark-results)
9. [Repository Structure](#repository-structure)
10. [License & Ethics Disclaimer](#license--ethics-disclaimer)

---

## Problem Context — The Defender's Dilemma

Traditional Network Intrusion Detection Systems (NIDS) and classic machine-learning flow classifiers suffer from structural limitations:

1. **Per-Flow Isolation**: Analyzing packets or flows in isolation misses multi-stage Advanced Persistent Threats (APTs) where individual actions look benign.
2. **Post-Hoc Alerting**: Alerts fire concurrent with or after compromise has occurred. By the time an alarm triggers, lateral movement or exfiltration is already underway.
3. **Zero Actionable Lead Time**: Defenders receive point-in-time alerts without trajectory forecasting, leading to alert fatigue and delayed triage.

### The Solution: Temporal World Models for Proactive Defence

SENTINEL-WM reframes intrusion detection from static classification to **temporal dynamical system forecasting**:
- **Continuous 10-Second State Windows**: Aggregates bidirectional flows into dense 53-dimensional state vectors capturing volume dynamics, flag distributions, port entropy, and inter-arrival timing.
- **Learned State Transitions**: Models latent network dynamics with a probabilistic transition network:
  $$\mu, \log \sigma = \text{StateTransitionNet}(z_t), \quad z_{t+1} \sim \mathcal{N}(\mu, \sigma)$$
- **Forward Trajectory Simulation**: Simulates $K = 6$ steps into the future (+10 s, +20 s, ..., +60 s) to forecast attack probabilities, confidence intervals, and progression states (`NORMAL` $\to$ `PRE_ATTACK` $\to$ `ONSET` $\to$ `ACTIVE` $\to$ `CONTINUATION`).
- **Deterministic Defence-in-Depth**: Pairs neural forward simulation with a deterministic **Custom Rules Engine** for policy enforcement and compliance verification.

---

## Key Advantages & Capabilities

| Capability | Traditional ML / Signature NIDS | SENTINEL-WM Platform |
|---|---|---|
| **Unit of Analysis** | Single flow or packet | Evolving 10-second network state window |
| **Temporal Modeling** | None (stateless) or sliding window | Probabilistic latent World Model ($P(S_{t+1} \mid S_{t-L:t})$) |
| **Prediction Horizon** | Past (what already happened) | **Future (+10 s to +60 s forward simulation)** |
| **Lead Time Metric** | 0 s (concurrent / retroactive) | **10 to 50+ seconds of proactive intervention lead time** |
| **ATT&CK Alignment** | Ad-hoc post-alert tagging | Real-time kill chain phase trajectory forecasting |
| **Wireshark Integration** | External manual review | **Native PCAP/PCAPNG ingestion with Berkeley Packet Filtering (BPF)** |
| **Live Sniffing** | Heavy sensor appliances | **Lightweight host capture agent with Npcap / Scapy** |
| **Hybrid Detection** | Black-box model or rigid rules | **Dual-Tier: Learned World Model + Custom Security Rules** |
| **Explainability** | Static global feature importance | Dynamic temporal saliency + per-anchor driving features |

---

## End-to-End System Architecture

```
                                  DATA INGESTION LAYER
   ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
   │    Wireshark PCAP       │   CICFlowMeter CSV      │    Live Network NIC     │
   │ (.pcap/.pcapng + BPF)   │    (Batch Datasets)     │  (Npcap Capture Agent)  │
   └────────────┬────────────┴────────────┬────────────┴────────────┬────────────┘
                │                         │                         │
                ▼                         ▼                         ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                      FASTAPI INFERENCE BACKEND (:8000)                      │
   │                                                                             │
   │  1. Flow Extraction & Normalization (5-tuple grouping, TCP/UDP reassembly) │
   │  2. Temporal Windowing (10s state windows, 53-dim metric representation)    │
   │  3. RobustScaler Normalization (fit on non-leaking operational baselines)   │
   │                                                                             │
   │  ┌───────────────────────────────────────────────────────────────────────┐  │
   │  │                       HYBRID INFERENCE ENGINE                         │  │
   │  │                                                                       │  │
   │  │   [ Component A: Sentinel World Model ]                               │  │
   │  │   GRU Encoder ──► z_t [160] ──► StateTransitionNet (μ, σ)             │  │
   │  │   └── K=6 Monte Carlo Latent Rollout (+10s ... +60s Horizon)          │  │
   │  │                                                                       │  │
   │  │   [ Component B: Neural Ensemble Baselines ]                          │  │
   │  │   TCN + LSTM + GRU Sequence Now-Casters (Calibrated Blend)            │  │
   │  │                                                                       │  │
   │  │   [ Component C: Custom Security & Heuristic Rules Engine ]           │  │
   │  │   SYN flood checks, port sweep entropy, exfiltration thresholds,      │  │
   │  │   and predictive onset early warning triggers                         │  │
   │  └───────────────────────────────────────────────────────────────────────┘  │
   │                                                                             │
   │  4. Horizon Attribution, Temporal Saliency & MITRE ATT&CK Phase Mapping     │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │  REST & WebSocket Streams
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                   SOC ANALYST CONSOLE (:3000 / :8080)                       │
   │                                                                             │
   │  • Multi-Horizon Predictive Curves (+10s ... +60s) with 95% Confidence Band │
   │  • Kill-Chain Progression Ribbon & MITRE ATT&CK Tactical Heatmap            │
   │  • Triggered Security Rules Drawer & Real-Time Incident Response Actions    │
   │  • Live Packet Telemetry Monitor & Wireshark Interface Capture Controller   │
   │  • Interactive Driving Feature Importance & Temporal Saliency Inspection    │
   └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites & System Requirements

### Operating Systems
- **Windows**: Windows 10 or Windows 11 (64-bit)
- **Linux**: Ubuntu 20.04+, Debian 11+, RHEL 8+, or Fedora
- **macOS**: macOS 12 Monterey or newer (Apple Silicon & Intel)

### Core Dependencies
- **Python**: Version `3.10` or `3.11` (with `pip` and `venv`)
- **Node.js**: Version `18.x` or `20.x` LTS (with `npm`)
- **Network Drivers (for Live Capture)**:
  - *Windows*: [Npcap](https://npcap.com/) (select **"Install Npcap in WinPcap API-compatible Mode"** during setup).
  - *Linux*: `libpcap-dev` (`sudo apt-get install libpcap-dev`).
- **Optional Utilities**:
  - [Wireshark](https://www.wireshark.org/) / `tshark` (for deep packet inspection and offline capture slicing).
  - [Docker](https://www.docker.com/) & Docker Compose (for containerized deployment).

---

## Installation & Setup Guide

### Option A: Local Development Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/rajesh08114/SIH_2k26.git
cd SIH_2k26
```

#### 2. Configure the Backend (FastAPI Service)
The backend includes a pre-packaged portable model bundle (`backend/models/`), so you can serve forecasts immediately without retraining:

```bash
cd backend
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
```

Start the inference server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*The API is now running at `http://localhost:8000`. Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.*

#### 3. Configure the Frontend (Next.js SOC Console)
Open a new terminal window:

```bash
cd frontend
npm install
npm run dev
```
*The SOC console is now running at `http://localhost:3000`.*

#### 4. Configure the Capture Agent (Optional — for Live NIC Capture)
To capture packets live from your machine's network card and stream them to the backend in real time:

```bash
cd capture-agent

# Requires elevated privileges (Run terminal as Administrator on Windows or sudo on Linux)
python -m venv .venv
# Activate environment...
pip install -e .

sentinel-capture run --backend ws://localhost:8000
```

---

### Option B: Containerized Deployment (Docker Compose)

You can launch the complete backend and static frontend console with a single command:

```bash
docker compose up --build
```
- **SOC Analyst Console**: `http://localhost:8080`
- **FastAPI Inference Backend**: `http://localhost:8000`

---

## How to Use the Platform

### 1. Instant Demo Flow Evaluation
To test the full predictive pipeline in under 10 seconds without any external files:
1. Open the console at `http://localhost:3000`.
2. Navigate to **Data Sources** (`/sources`).
3. Under the **CSV Upload** tab, click **"Use sample demo slice"** (selects `demo/demo_dos_onset.csv` or `demo_benign.csv`).
4. Click **Run Forecast**.
5. The console will reassemble state windows, simulate 6 forward horizons, and redirect you to the **SOC Dashboard** showing lead times, progression ribbons, and driving features.

---

### 2. Wireshark PCAP / PCAPNG Ingestion with BPF Filtering
SENTINEL-WM natively parses `.pcap` and `.pcapng` traces captured from Wireshark or tcpdump:
1. In the console, go to **Data Sources** $\to$ **PCAP Upload**.
2. Drag and drop your `.pcap` or `.pcapng` file.
3. *(Optional)* Apply a **Wireshark BPF Capture Filter** to isolate relevant traffic:
   - Click a preset chip (e.g. `HTTP/HTTPS`, `DNS`, `Exclude Broadcast/Multicast`, or `Admin Ports`) or enter custom BPF syntax (e.g., `tcp port 80 or ip host 192.168.1.50`).
4. Click **Run Forecast**. The backend parses packet timestamps, reassembles bidirectional TCP/UDP flows via `FlowMeter`, constructs 10-second windows, and performs forward rollouts.

---

### 3. Live Network Interface Sniffing (Host Capture Agent)
To monitor local physical or virtual network interfaces in real time:
1. Ensure the `capture-agent` is running on your host (`sentinel-capture run --backend ws://localhost:8000`).
2. In the console, navigate to **Live Telemetry** (`/live`).
3. Select **Live Capture**.
4. The interface picker will automatically list all detected network adapters (e.g. Wi-Fi, Ethernet, virtual adapters).
5. Select the network card you wish to monitor and optionally apply a BPF filter preset.
6. Click **Start Live Capture**. The console will begin streaming live 10-second state windows, updating predictive horizon curves every 10 seconds.

---

### 4. Custom Security & Heuristic Rules Engine
In addition to the Learned World Model, SENTINEL-WM evaluates a **deterministic security rules engine** on every state window for defense-in-depth:

- **Built-in Rules**:
  - `RULE-SEC-001` (**Aggressive TCP SYN Scan**): Detects high concentrations of half-open SYN packets (`syn_ratio >= 0.70`).
  - `RULE-SEC-002` (**Network Port Sweep**): Flags elevated destination port entropy (`port_entropy >= 0.85`).
  - `RULE-SEC-003` (**High-Volume Data Velocity**): Alerts on outbound data transfer surges exceeding operational baseline throughput.
  - `RULE-AI-004` (**Predictive Attack Onset Trigger**): Triggers when the World Model projects trajectory transition into `ONSET` or `ACTIVE` with $P(\text{attack}) \ge 0.70$.
  - `RULE-AI-005` (**Imminent Critical Trajectory**): High-certainty alarm when lead time is $\le 20\text{s}$ and forecast probability exceeds $0.85$.

- **Rule Management**:
  - View all active rules via `GET /rules`.
  - Dynamically toggle rules on/off via `POST /rules/toggle` or through the SOC console.
  - Triggered rules appear directly in the **Anchor Table** and in the dedicated **Triggered Security Rules** drawer.

---

### 5. SOC Analyst Console Walkthrough
The console provides five purpose-built views for cyber incident response:
- **SOC Dashboard** (`/dashboard`): High-level KPI tiles (Alert Windows, Peak Attack Probability, Earliest Lead Time, ATT&CK Phases), interactive Anchor Table, Multi-Horizon Rollout Curves, and Feature Attribution charts.
- **Kill-Chain Progression Ribbon**: Visualizes the progression state over the next 60 seconds (`NORMAL` $\to$ `PRE_ATTACK` $\to$ `ONSET` $\to$ `ACTIVE` $\to$ `CONTINUATION`).
- **MITRE ATT&CK Phase Timeline**: Maps projected tactics across horizons (Reconnaissance, Initial Access, Lateral Movement, Exfiltration) with confidence scoring.
- **Live Telemetry Monitor** (`/live`): Real-time streaming radar and rolling probability timeline for continuous monitoring.
- **System Architecture & Model Card** (`/architecture`, `/model`): Comprehensive breakdown of model layers, weights, calibration curves, and feature schemas.

---

## Inference Engine & API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/forecast/csv` | Ingests a CICFlowMeter flow CSV $\to$ returns multi-horizon forecasts. |
| `POST` | `/forecast/pcap` | Ingests `.pcap` / `.pcapng` capture with optional `bpf_filter` $\to$ reassembles flows $\to$ returns forecasts. |
| `GET` | `/rules` | Lists all active deterministic security detection rules and metadata. |
| `POST` | `/rules/toggle` | Enables or disables a specific detection rule (`{ rule_id, enabled }`). |
| `POST` | `/live/sessions` | Spawns a live capture or synthetic test-bed forecasting session. |
| `GET` | `/live/sessions` | Lists active streaming sessions and their subscriber counts. |
| `WS` | `/live/sessions/{id}/stream` | WebSocket subscription streaming live 10-second forecast anchors. |
| `WS` | `/agent` | WebSocket channel where host packet capture agents connect. |
| `GET` | `/agent/status` | Reports connected capture agents and enumerated network interfaces. |
| `GET` | `/meta` | Returns model bundle metadata, feature dimensions, and operating thresholds. |
| `GET` | `/health` | Service health, uptime, and loaded model bundle status. |

---

## Empirical Benchmark Results

Evaluated on the leakage-safe `stratified` split across all 5 days of the standard **CIC-IDS-2017** benchmark (~2.8 million flows, 15 attack families):

| Model | PR-AUC | F1 Score | AUROC | Horizon Decay | Key Capabilities |
|---|---|---|---|---|---|
| **SENTINEL-WM (System)** | **0.992** | **0.968** | **1.000** | **Flattest (maintains accuracy across K=6)** | **World Model rollouts + baselines + lead time + ATT&CK** |
| LSTM Baseline | 0.992 | 0.968 | 1.000 | Moderate decay | Static point-in-time sequence classification |
| GRU Baseline | 0.984 | 0.961 | 1.000 | Moderate decay | Compact recurrent sequence baseline |
| TCN Baseline | 0.981 | 0.932 | 1.000 | Moderate decay | Dilated temporal convolutions |
| Raw World Model | 0.957 | 0.909 | 0.998 | Stable rollout | Pure forward simulation without ensemble blend |

### Actionable Lead Time Metrics
- **Mean Lead Time**: **35 to 50 seconds** prior to attack onset on sustained kill chains.
- **False Alarm Rate (FAR)**: Calibrated at $< 0.8\%$ on benign operational traffic.
- **Rollout Consistency**: Monte Carlo variance ($\sigma$) serves as an intrinsic uncertainty metric, alerting analysts when network trajectory stability degrades.

---

## Repository Structure

```
SIH_2k26/
├── backend/                  # FastAPI inference service & serving runtime
│   ├── app/
│   │   ├── api/              # REST & WebSocket API route controllers
│   │   ├── rules/            # Custom Security Detection Rules Engine
│   │   ├── live/             # Real-time streaming sessions & ring buffers
│   │   ├── agent/            # Capture agent registry & WebSocket handler
│   │   ├── sentinel_infer/   # Self-contained model runtime & PCAP reassembly
│   │   └── main.py           # Application entrypoint & lifespan manager
│   ├── models/               # Shipped model bundle (world_model.pt, scalers)
│   └── tests/                # Automated backend test suite (pytest)
│
├── frontend/                 # Next.js 14 SOC Analyst Console
│   ├── app/                  # App router pages (dashboard, live, sources, etc.)
│   ├── components/
│   │   ├── charts/           # Horizon forecast charts & saliency heatmaps
│   │   ├── panels/           # Anchor table, progression ribbons, rules drawer
│   │   ├── live/             # Network interface picker & scenario form
│   │   └── sources/          # Wireshark PCAP upload & CSV wizard
│   └── lib/                  # State store (Zustand), API client, and schemas
│
├── capture-agent/            # Host-level live packet sniffer
│   └── sentinel_capture/     # Npcap / Scapy NIC enumeration & flowmeter
│
├── research/                 # Machine learning research & training package
│   └── sentinel_wm/          # Preprocessing, world models, training, evaluation
│
├── demo/                     # Real CIC-IDS-2017 traffic slices for zero-setup demo
├── docs/                     # Technical specifications, architecture, and guides
├── docker-compose.yml        # Multi-container local deployment spec
└── README.md                 # Project documentation
```

---

## License & Ethics Disclaimer

- **License**: Released under the [MIT License](LICENSE).
- **Academic & Research Notice**: This software is an open research prototype developed for anticipatory cyber defense. Datasets utilized for benchmarking (CIC-IDS-2017) are the property of their respective publishers (Canadian Institute for Cybersecurity) and are not redistributed within this repository.
- **Safe Operations**: The live capture agent passively sniffs network traffic and does not inject packets or disrupt operational network infrastructure.
