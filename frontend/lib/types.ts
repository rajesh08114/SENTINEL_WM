// Shapes returned by the SENTINEL-WM backend. `zod` schemas in api.ts parse the
// wire payloads; these types are the parsed results.

export interface AttckAssessment {
  progression_state?: string;
  mitre_tactic: string;
  technique_ids: string[];
  kill_chain_phase: string;
  confidence: "High" | "Medium" | "Low" | string;
  rationale?: string;
  dominant_family?: string;
  family_transition?: string | null;
  family_inferred?: boolean;
  possible_family?: string | null;
  possible_family_confidence?: string | null;
}

export interface HorizonStep {
  k: number;
  horizon_seconds: number;
  attack_prob: number;
  attack_ci: [number, number];
  attack_std?: number;
  detection_prob?: number;
  progression_state: string;
  progression_dist: Record<string, number>;
  attck: AttckAssessment;
}

export interface DrivingFeatures {
  top_features: { feature: string; importance: number; signed?: number }[];
  temporal_saliency_gradient?: number[];
  temporal_saliency_attention?: number[];
}

export interface AnchorForecast {
  alert: boolean;
  first_alert_k: number | null;
  lead_time_seconds: number | null;
  max_attack_prob: number;
  max_detection_prob?: number;
  detection_model?: string;
  horizon: HorizonStep[];
  driving_features?: DrivingFeatures | null;
  meta?: Record<string, unknown> & { window_index?: number; stream_window?: number };
}

export interface ForecastMeta {
  n_flows: number;
  n_windows: number;
  n_anchors: number;
  window_seconds: number;
  history_windows: number;
  horizon_steps: number;
  feature_dim: number;
  model?: string;
  blend_weight?: number | null;
  family_hint?: string | null;
}

export interface ForecastSummary {
  n_alerts: number;
  max_attack_prob: number;
  phases: (string | { mitre_tactic?: string; kill_chain_phase?: string })[];
  alert_threshold: number;
}

export interface ForecastResponse {
  meta: ForecastMeta;
  summary: ForecastSummary;
  anchors: AnchorForecast[];
}

export interface MetaResponse {
  history_windows: number;
  horizon_steps: number;
  feature_dim: number;
  window_seconds: number;
  progression_states: string[];
  feature_names: string[];
  alert_threshold: number;
  device: string;
  serve_mode: string;
  blend_members: string[];
  blend_weight: number | null;
  bundle: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  detail?: string | null;
  bundle?: Record<string, unknown>;
  live?: { sessions: number; max: number };
  agent?: { connected: boolean; count: number };
  uptime_s?: number;
}

export interface JobStatus {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  error?: string | null;
  created_at: string;
  finished_at?: string | null;
  has_result: boolean;
}

export interface LiveSessionInfo {
  id: string;
  source_kind: "synthetic" | "capture" | string;
  state: "starting" | "running" | "stopping" | "stopped" | "error" | string;
  error: string | null;
  params: Record<string, unknown>;
  stats: { flows_in: number; windows: number; forecasts: number; alerts: number };
  created_at: number;
  last_activity: number;
  subscribers: number;
}

export interface NetInterface {
  name: string;
  description: string;
  ipv4: string | null;
  netmask: string | null;
  mac: string | null;
  is_up: boolean;
  is_loopback: boolean;
}

export interface AgentStatus {
  connected: boolean;
  count: number;
  agents: { name: string; interfaces: NetInterface[]; session_id: string | null }[];
}

export type LiveFrame =
  | ({ type: "status" } & LiveSessionInfo)
  | ({ type: "forecast" } & AnchorForecast)
  | { type: "error"; detail: string }
  | { type: "bye" };
