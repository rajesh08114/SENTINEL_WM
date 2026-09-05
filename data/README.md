# `data/` — input datasets

This folder holds the **labelled unified flow CSVs** the pipeline consumes.
Nothing here is committed to git (see `../.gitignore`) — the files are large and
CIC-IDS-2017 has its own redistribution terms.

## What goes here

| File | Produced by | Size |
|---|---|---|
| `unified_<Day>-WorkingHours_labeled.csv` | `extraction/label_mapping.ipynb` | ~0.1–2 GB / day |

Each row is one bidirectional flow with:
* Tier 1 — CICFlowMeter-style flow features (from the CIC-IDS-2017 CSV),
* Tier 2 — packet-level features (TTL/window/payload moments, retransmission
  count, port-scan entropy) computed from the matching PCAP by
  `extraction/extractor.py`,
* `flow_start_epoch`, `source_file`, and a mapped `Label`.

The repo ships with `unified_Wednesday-WorkingHours_labeled.csv` already built.

## Adding more CIC-IDS-2017 days (recommended)

1. Download the CIC-IDS-2017 PCAP + `MachineLearningCVE` / `GeneratedLabelledFlows`
   CSVs from the University of New Brunswick
   (<https://www.unb.ca/cic/datasets/ids-2017.html>).
2. Run `extraction/extractor.py` on the day's `.pcap` to get the unified feature
   parquet/CSV, then `extraction/label_mapping.ipynb` to transfer the official
   labels → `data/unified_<Day>-WorkingHours_labeled.csv`.
3. Add the new path to `RAW_FLOW_CSVS` in `sentinel_wm/config.py`.
4. Re-run `python -m sentinel_wm.cli all`.

Day schedule (`sentinel_wm/config.py:DAY_SCHEDULE`):

| Day | Attacks | Split role once present |
|---|---|---|
| Monday | benign only | train |
| Tuesday | FTP/SSH-Patator | train |
| Wednesday | DoS ×4, Heartbleed | train |
| Thursday | Web attacks, **Infiltration** | validation |
| Friday | Botnet, PortScan, DDoS | test |

With ≥2 days present, `SplitConfig.mode="auto"` switches from the single-day
block-interleaved split to this proper day-based split automatically.

## Cross-dataset evaluation (optional)

`CTU-13` (NetFlow) and `UNSW-NB15` (CSV) can be dropped here for
out-of-distribution testing once a column adapter is written — not part of the
current build.
