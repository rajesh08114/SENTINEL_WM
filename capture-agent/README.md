# sentinel-capture — live network-capture agent

Sniffs a local network interface, reassembles packets into CICFlowMeter-schema
**bidirectional flow rows**, and streams them to the SENTINEL-WM backend over
`WS /agent` for real-time attack-progression forecasting.

It runs on the **host**, not in a container — live capture needs raw-socket
privileges and a packet-capture driver.

## Prerequisites

- **Windows:** install [Npcap](https://npcap.com/) (tick *"WinPcap API-compatible
  mode"*). Run the agent from an **Administrator** PowerShell.
- **Linux/macOS:** run with `sudo`, or grant `cap_net_raw` to the Python binary.
- Huawei **eNSP**: its cloud/bridge devices attach to VirtualBox host-only
  adapters, which show up in the interface list like any other NIC — pick the one
  carrying your topology's traffic.

## Install

```bash
cd capture-agent
python -m venv .venv && .venv\Scripts\activate      # or source .venv/bin/activate
pip install -e .
```

## Use

```bash
# list interfaces (name, description, IPv4)
sentinel-capture interfaces

# connect to the backend and wait for a capture command from the console
sentinel-capture run --backend ws://localhost:8000 --name my-host
```

Then in the SENTINEL-WM console: **Live → Live capture**, pick the interface,
optionally set an IP/CIDR filter (e.g. `host 10.0.0.5`, `net 192.168.56.0/24`),
and **Start**. Forecasts stream to the dashboard.

## What it sends

Per flow, when it closes (FIN/RST) or times out: the 11 required CICFlowMeter
columns + `Source Port` + `flag_true_{fin,syn,rst,psh,ack,urg}` +
`Fwd/Bwd IAT Total` + `pkt_len_mean`. Everything else the model wants defaults
to 0, exactly as an uploaded CSV would.

## Tests

```bash
pytest            # flowmeter + interface enumeration; no live capture needed
```
