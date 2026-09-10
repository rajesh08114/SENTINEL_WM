// Static research artifacts — the leakage-safe benchmark (runs/benchmarks/benchmark.md,
// git_sha 76c55e1). Headline system metrics are also fetched live from /models.

export const SPLIT_NOTE =
  "Test split = Friday (2761 sequences, 2.8% attack). Threshold = best-F1 on validation " +
  "within an FPR ≤ 15% budget, per model. With ~76 positive test sequences the top cluster " +
  "is within noise — PR-AUC / F1* / AUROC are the trustworthy ranks.";

export interface BenchRow {
  model: string;
  family: string;
  f1: number;
  f1star: number;
  prauc: number;
  auroc: number;
  progAcc: number | null;
  params: number | null;
  inferMs: number;
  tier: "system" | "member" | "reference" | "other";
}

export const SCOREBOARD: BenchRow[] = [
  { model: "SENTINEL-WM (system)", family: "system", f1: 0.920, f1star: 0.968, prauc: 0.992, auroc: 1.000, progAcc: 0.980, params: 1_143_542, inferMs: 0.232, tier: "system" },
  { model: "LSTM", family: "nn", f1: 0.968, f1star: 0.968, prauc: 0.992, auroc: 1.000, progAcc: 0.989, params: 247_716, inferMs: 0.038, tier: "member" },
  { model: "persistence", family: "reference", f1: 0.993, f1star: 0.993, prauc: 0.987, auroc: 1.000, progAcc: null, params: null, inferMs: 0.0, tier: "reference" },
  { model: "TCN", family: "nn", f1: 0.874, f1star: 0.933, prauc: 0.981, auroc: 0.999, progAcc: 0.991, params: 172_644, inferMs: 0.063, tier: "member" },
  { model: "GRU", family: "nn", f1: 0.955, f1star: 0.955, prauc: 0.977, auroc: 1.000, progAcc: 0.994, params: 191_140, inferMs: 0.040, tier: "member" },
  { model: "SENTINEL-WM (world model only)", family: "world_model", f1: 0.828, f1star: 0.914, prauc: 0.960, auroc: 0.998, progAcc: 0.980, params: 532_042, inferMs: 0.232, tier: "other" },
  { model: "MLP", family: "nn", f1: 0.818, f1star: 0.827, prauc: 0.849, auroc: 0.986, progAcc: 0.985, params: 519_460, inferMs: 0.007, tier: "other" },
  { model: "GAT (graph)", family: "graph", f1: 0.591, f1star: 0.621, prauc: 0.599, auroc: 0.973, progAcc: 0.969, params: 246_820, inferMs: 1.652, tier: "other" },
  { model: "XGBoost (sequence)", family: "classical", f1: 0.427, f1star: 0.462, prauc: 0.446, auroc: 0.811, progAcc: null, params: null, inferMs: 0.013, tier: "other" },
  { model: "Random Forest (sequence)", family: "classical", f1: 0.440, f1star: 0.448, prauc: 0.355, auroc: 0.397, progAcc: null, params: null, inferMs: 0.171, tier: "other" },
];

export const HORIZON_F1: { model: string; f1: number[] }[] = [
  { model: "SENTINEL-WM (system)", f1: [0.932, 0.961, 0.948, 0.935, 0.908, 0.889] },
  { model: "LSTM", f1: [0.962, 0.962, 0.948, 0.947, 0.917, 0.859] },
  { model: "GRU", f1: [0.955, 0.954, 0.941, 0.919, 0.905, 0.904] },
  { model: "TCN", f1: [0.879, 0.873, 0.871, 0.864, 0.850, 0.832] },
  { model: "world model only", f1: [0.836, 0.826, 0.840, 0.852, 0.855, 0.829] },
  { model: "persistence", f1: [0.994, 0.987, 0.980, 0.973, 0.966, 0.952] },
];

export const PARAM_BUDGET: { part: string; params: number; note: string }[] = [
  { part: "Bi-GRU encoder", params: 232_320, note: "2 × GRU(d_model=160), bidirectional" },
  { part: "Additive attention pool", params: 103_040, note: "L→1 context vector; attention-saliency source" },
  { part: "State-Transition Network", params: 103_680, note: "MLP → (μ, logσ); reparameterised sampling" },
  { part: "Heads (attack + progression + next-state)", params: 83_562, note: "3 linear heads on z" },
  { part: "Elapsed-time positional encoding", params: 9_440, note: "sinusoidal(Δt) → d_model" },
  { part: "world model subtotal", params: 532_042, note: "" },
  { part: "TCN member", params: 172_644, note: "dilated causal conv — decorrelated P(attack)" },
  { part: "LSTM member", params: 247_716, note: "recurrent P(attack)" },
  { part: "GRU member", params: 191_140, note: "recurrent P(attack)" },
  { part: "SENTINEL-WM (system) total", params: 1_143_542, note: "world model + 3 blend members" },
];

export const LEAKAGE_CONTROLS: { control: string; what: string }[] = [
  { control: "Contiguous benign backbone", what: "each day's benign windows split 60/20/20 by wall-clock time, never shuffled — a train window is always earlier than the val/test windows around it." },
  { control: "Whole-episode assignment", what: "an attack episode is assigned entirely to one split, rotating per family (train, val, train, test). No episode is half-seen." },
  { control: "Long-episode internal cut", what: "episodes ≥ 54 windows get an internal 60/20/20 cut with a purge band so a very long DoS still contributes to all splits without leaking across the cut." },
  { control: "Boundary-sequence purge", what: "any L+K sequence whose [t−11 … t+6] span crosses a split boundary is dropped (SequenceConfig.purge_boundary_sequences)." },
  { control: "Span-purity assertion", what: "build_sequences re-derives a split map per window and asserts every emitted sequence is single-split — the pipeline fails loudly if leakage slips in." },
  { control: "Train-only statistics", what: "winsorisation percentiles and the RobustScaler are fit on DAY_SCHEDULE train days only; val/test are transformed, never fit." },
  { control: "Best-F1 threshold on validation", what: "the alert threshold is the best-F1 point on the val split within an FPR budget — never tuned on test. (An earlier FPR-only rule collapsed to all-positive on the prevalence-shifted test set.)" },
];

export const TRUST_POINTS: string[] = [
  "The top cluster — system, LSTM, GRU, persistence — is within noise of a ~76-positive test set. Treat PR-AUC / F1* / AUROC as the ranks; a 0.02 F1 gap there is not real.",
  "`persistence` (predict the last observed label) tops raw F1 because the Friday attacks are long and contiguous — it is the honest baseline the learned models must beat on *earliness*, not on nowcast F1.",
  "The benchmark measures now-casting (is the current window part of an attack that continues?), not cold onset forecasting. Mean-lead-time ≈ 0 s is a split artefact — the attacks don't have clean benign→attack transitions to lead.",
  "Four rare families (Bot, DoS Hulk on some days, Heartbleed, Infiltration) have < 8 positive test sequences and are excluded from per-family metrics.",
  "Synthetic test-bed traffic is out-of-distribution — the model runs hot there and absolute P(attack) is not calibrated. Use real flow CSVs or live capture for calibrated numbers.",
];
