import { z } from "zod";
import type {
  AgentStatus, ForecastResponse, HealthResponse, JobStatus,
  LiveSessionInfo, MetaResponse,
} from "./types";

const LS_KEY = "sentinel.apiBase";
const ENV_BASE =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE) || "";

export function apiBase(): string {
  if (typeof window !== "undefined") {
    const q = new URLSearchParams(window.location.search).get("api");
    if (q) return q.replace(/\/$/, "");
    try {
      const v = window.localStorage.getItem(LS_KEY);
      if (v) return v.replace(/\/$/, "");
    } catch {
      /* ignore */
    }
    if (!ENV_BASE) return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return (ENV_BASE || "http://localhost:8000").replace(/\/$/, "");
}

export function setApiBase(url: string) {
  try {
    window.localStorage.setItem(LS_KEY, url.replace(/\/$/, ""));
  } catch {
    /* ignore */
  }
}

export function wsBase(): string {
  return apiBase().replace(/^http/, "ws");
}

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function req<T>(path: string, init?: RequestInit, parse?: (d: unknown) => T): Promise<T> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 15_000);
  let r: Response;
  try {
    r = await fetch(apiBase() + path, { ...init, signal: ac.signal });
  } catch (e) {
    throw new ApiError(
      (e as Error)?.name === "AbortError"
        ? `backend timed out (${path})`
        : `cannot reach backend at ${apiBase()}`,
      0,
      e
    );
  } finally {
    clearTimeout(timer);
  }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail =
      (body && typeof body === "object" && "detail" in body && (body as any).detail) ||
      `${r.status} ${r.statusText}`;
    throw new ApiError(String(detail), r.status, body);
  }
  return parse ? parse(body) : (body as T);
}

/* ------------------------------- zod schemas ------------------------------ */
const zHorizon = z
  .object({
    k: z.number(),
    horizon_seconds: z.number(),
    attack_prob: z.number(),
    attack_ci: z.tuple([z.number(), z.number()]),
    detection_prob: z.number().optional(),
    progression_state: z.string(),
    progression_dist: z.record(z.number()),
    attck: z.object({ mitre_tactic: z.string(), kill_chain_phase: z.string() }).passthrough(),
  })
  .passthrough();

const zAnchor = z
  .object({
    alert: z.boolean(),
    first_alert_k: z.number().nullable(),
    lead_time_seconds: z.number().nullable(),
    max_attack_prob: z.number(),
    horizon: z.array(zHorizon).length(6),
  })
  .passthrough();

const zForecast = z
  .object({
    meta: z.object({ n_flows: z.number(), n_anchors: z.number() }).passthrough(),
    summary: z.object({ n_alerts: z.number(), alert_threshold: z.number() }).passthrough(),
    anchors: z.array(zAnchor),
  })
  .passthrough();

const zMeta = z
  .object({
    history_windows: z.number(),
    horizon_steps: z.number(),
    feature_dim: z.number(),
    window_seconds: z.number(),
    progression_states: z.array(z.string()),
    feature_names: z.array(z.string()),
    alert_threshold: z.number(),
    device: z.string(),
    serve_mode: z.string(),
    blend_members: z.array(z.string()).default([]),
    blend_weight: z.number().nullable().default(null),
    bundle: z.record(z.unknown()).default({}),
  })
  .passthrough();

const zLive = z
  .object({
    id: z.string(),
    source_kind: z.string(),
    state: z.string(),
    error: z.string().nullable(),
    params: z.record(z.unknown()),
    stats: z.object({
      flows_in: z.number(), windows: z.number(),
      forecasts: z.number(), alerts: z.number(),
    }),
    created_at: z.number(),
    last_activity: z.number(),
    subscribers: z.number(),
  })
  .passthrough();

const zAgent = z
  .object({
    connected: z.boolean(),
    count: z.number(),
    agents: z.array(
      z.object({
        name: z.string(),
        interfaces: z.array(z.record(z.unknown())),
        session_id: z.string().nullable(),
      })
    ),
  })
  .passthrough();

/* --------------------------------- calls -------------------------------- */
export const api = {
  health: () => req<HealthResponse>("/health"),
  meta: () => req<MetaResponse>("/meta", undefined, (d) => zMeta.parse(d) as unknown as MetaResponse),
  models: () => req<Record<string, unknown>>("/models"),
  agentStatus: () =>
    req<AgentStatus>("/agent/status", undefined, (d) => zAgent.parse(d) as unknown as AgentStatus),

  jobs: () => req<JobStatus[]>("/jobs"),
  job: (id: string) => req<JobStatus>(`/jobs/${id}`),
  jobResult: (id: string) =>
    req<ForecastResponse>(`/jobs/${id}/result`, undefined, (d) => zForecast.parse(d) as unknown as ForecastResponse),

  async forecastCsv(
    file: File | Blob,
    opts: { familyHint?: string; explain?: boolean } = {}
  ): Promise<{ job: true; job_id: string } | { job: false; result: ForecastResponse }> {
    const fd = new FormData();
    fd.append("file", file, (file as File).name || "flows.csv");
    if (opts.familyHint) fd.append("family_hint", opts.familyHint);
    fd.append("explain", String(opts.explain ?? true));
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 180_000); // 3 min ceiling
    let r: Response;
    try {
      r = await fetch(apiBase() + "/forecast/csv", {
        method: "POST",
        body: fd,
        signal: ac.signal,
      });
    } catch (e) {
      throw new ApiError(
        (e as Error)?.name === "AbortError"
          ? "backend did not respond within 3 min"
          : `cannot reach backend at ${apiBase()} — is it running?`,
        0,
        e
      );
    } finally {
      clearTimeout(timer);
    }
    const body = await r.json().catch(() => ({}));
    if (r.status === 202) return { job: true, job_id: (body as any).job_id };
    if (!r.ok) throw new ApiError(String((body as any).detail || r.status), r.status, body);
    return { job: false, result: zForecast.parse(body) as unknown as ForecastResponse };
  },

  async forecastPcap(
    file: File | Blob,
    opts: { familyHint?: string; explain?: boolean; bpfFilter?: string } = {}
  ): Promise<ForecastResponse> {
    const fd = new FormData();
    fd.append("file", file, (file as File).name || "capture.pcap");
    if (opts.familyHint) fd.append("family_hint", opts.familyHint);
    if (opts.bpfFilter) fd.append("bpf_filter", opts.bpfFilter);
    fd.append("explain", String(opts.explain ?? true));
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 240_000); // pcap parsing can be slow
    let r: Response;
    try {
      r = await fetch(apiBase() + "/forecast/pcap", {
        method: "POST",
        body: fd,
        signal: ac.signal,
      });

    } catch (e) {
      throw new ApiError(
        (e as Error)?.name === "AbortError"
          ? "backend did not finish parsing the pcap within 4 min"
          : `cannot reach backend at ${apiBase()} — is it running?`,
        0,
        e
      );
    } finally {
      clearTimeout(timer);
    }
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new ApiError(String((body as any).detail || r.status), r.status, body);
    return zForecast.parse(body) as unknown as ForecastResponse;
  },

  listLiveSessions: () =>
    req<LiveSessionInfo[]>("/live/sessions", undefined, (d) =>
      z.array(zLive).parse(d) as unknown as LiveSessionInfo[]
    ),
  getLiveSession: (id: string) =>
    req<LiveSessionInfo>(`/live/sessions/${id}`, undefined, (d) => zLive.parse(d) as unknown as LiveSessionInfo),
  createLiveSession: (payload: Record<string, unknown>) =>
    req<LiveSessionInfo>(
      "/live/sessions",
      { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) },
      (d) => zLive.parse(d) as unknown as LiveSessionInfo
    ),
  deleteLiveSession: (id: string) =>
    fetch(apiBase() + `/live/sessions/${id}`, { method: "DELETE" }).then((r) => r.ok),
};

export async function pollJob(
  id: string,
  onTick?: (s: JobStatus) => void,
  interval = 1500
): Promise<ForecastResponse> {
  for (;;) {
    const s = await api.job(id);
    onTick?.(s);
    if (s.status === "done") return api.jobResult(id);
    if (s.status === "error") throw new ApiError(s.error || "job failed", 500, s);
    await new Promise((r) => setTimeout(r, interval));
  }
}
