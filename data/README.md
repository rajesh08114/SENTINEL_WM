# `data/` — input datasets

This folder holds the **labelled unified flow CSVs** the pipeline consumes.
Nothing here is committed to git (see `../.gitignore`) — the files are large and
CIC-IDS-2017 has its own redistribution terms.

## What goes here

| File | Produced by | Size |
|---|---|---|
| `unified_AllDays_labeled.csv` | all 5 days concatenated (has a `source_day` column) | ~2.4 GB |
| `unified_<Day>-WorkingHours_labeled.csv` | `extraction/label_mapping.ipynb`, per day | ~0.1–2 GB / day |

`config.RAW_FLOW_CSVS` points at `unified_AllDays_labeled.csv`; if it is missing,
`RAW_FLOW_CSVS_FALLBACK` uses `unified_Wednesday-WorkingHours_labeled.csv`.

Each row is one bidirectional flow with:
* Tier 1 — CICFlowMeter-style flow features (from the CIC-IDS-2017 CSV),
* Tier 2 — packet-level features (TTL/window/payload moments, retransmission
  count, port-scan entropy) computed from the matching PCAP by
  `extraction/extractor.py`,
* `flow_start_epoch`, `source_file` / `source_day`, and a mapped `Label`
  (15 attack families across the 5 days).

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
| Monday | benign only |
| Tuesday | FTP/SSH-Patator |
| Wednesday | DoS ×4, Heartbleed |
| Thursday | Web attacks, **Infiltration** |
| Friday | Botnet, PortScan, DDoS |

The default `SplitConfig.mode="auto"` → **`stratified`** does *not* split by day:
it assigns whole attack episodes to train/val/test per family (leakage-safe) so
every family with ≥3 bursts is in all 3 splits. The day-based Mon-Wed / Thu / Fri
split above is available as the zero-shot secondary benchmark
(`--split day --outdir research_zeroshot`). See `RUN.md` §3.

## Cross-dataset evaluation (optional)

`CTU-13` (NetFlow) and `UNSW-NB15` (CSV) can be dropped here for
out-of-distribution testing once a column adapter is written — not part of the
current build.
