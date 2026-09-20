import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { AnchorForecast, ForecastResponse, LiveSessionInfo } from "./types";
import type { ColumnMatch } from "./schema";
import type { CsvSummary } from "./csv";
import type { ScenarioValues } from "@/components/live/ScenarioForm";
import { openLiveStream, type LiveStream } from "./ws";
import { api } from "./api";

export interface CsvState {
  file?: File;
  name: string;
  header: string[];
  rows: string[][];
  summary: CsvSummary;
  match: ColumnMatch;
}

export type ActiveSource = "live" | "upload";
export type LiveSourceMode = "synthetic" | "capture";

interface Store {
  // Active view preference
  activeSource: ActiveSource;
  setActiveSource: (src: ActiveSource) => void;

  // Uploaded dataset forecast & inspection
  csv: CsvState | null;
  result: ForecastResponse | null;
  selectedAnchor: number;
  setCsv: (c: CsvState | null) => void;
  setResult: (r: ForecastResponse | null) => void;
  clearResult: () => void;
  selectAnchor: (i: number) => void;

  // Live session state (persists across navigation)
  liveMode: LiveSourceMode;
  setLiveMode: (m: LiveSourceMode) => void;
  scenarioForm: ScenarioValues;
  setScenarioForm: (f: ScenarioValues) => void;
  captureIface: string;
  captureBpf: string;
  setCaptureParams: (iface: string, bpf: string) => void;

  liveSessionId: string | null;
  liveStatus: LiveSessionInfo | null;
  liveForecasts: AnchorForecast[];
  liveConnected: boolean;
  livePaused: boolean;

  // Live session actions (singleton stream management)
  startLiveSession: (session: LiveSessionInfo) => void;
  attachLiveStream: (sessionId: string) => void;
  stopLiveSession: () => Promise<void>;
  toggleLivePause: () => void;
  pushLiveForecast: (f: AnchorForecast) => void;
  resetLive: () => void;
  reconnectLiveIfActive: () => Promise<void>;
}

// Global singleton live stream reference so navigation never terminates the connection
let globalLiveStream: LiveStream | null = null;
let currentConnectedSessionId: string | null = null;

export const useStore = create<Store>()(
  persist(
    (set, get) => ({
      activeSource: "live",
      setActiveSource: (src) => set({ activeSource: src }),

      csv: null,
      result: null,
      selectedAnchor: 0,
      setCsv: (c) => set({ csv: c }),
      setResult: (r) => set({ result: r, selectedAnchor: 0, activeSource: "upload" }),
      clearResult: () => set({ result: null }),
      selectAnchor: (i) => set({ selectedAnchor: i }),

      liveMode: "synthetic",
      setLiveMode: (m) => set({ liveMode: m }),
      scenarioForm: {
        scenario: "portscan",
        rate: 25,
        duration_s: 600,
        speed: 30,
        seed: 1,
      },
      setScenarioForm: (f) => set({ scenarioForm: f }),
      captureIface: "",
      captureBpf: "",
      setCaptureParams: (iface, bpf) => set({ captureIface: iface, captureBpf: bpf }),

      liveSessionId: null,
      liveStatus: null,
      liveForecasts: [],
      liveConnected: false,
      livePaused: false,

      startLiveSession: (session) => {
        if (globalLiveStream) {
          globalLiveStream.close();
          globalLiveStream = null;
          currentConnectedSessionId = null;
        }

        set({
          liveSessionId: session.id,
          liveStatus: session,
          liveForecasts: [],
          liveConnected: false,
          livePaused: false,
          activeSource: "live",
        });

        currentConnectedSessionId = session.id;
        globalLiveStream = openLiveStream(session.id, {
          onOpen: () => set({ liveConnected: true }),
          onClose: () => set({ liveConnected: false }),
          onStatus: (st) => set({ liveStatus: st }),
          onForecast: (f) => {
            if (get().livePaused) return;
            set((st) => ({
              liveForecasts: [...st.liveForecasts.slice(-199), f],
            }));
          },
          onError: (err) => {
            console.warn("[SENTINEL live stream error]", err);
          },
        });
      },

      attachLiveStream: (sessionId) => {
        if (currentConnectedSessionId === sessionId && globalLiveStream) {
          return; // Already attached and streaming
        }

        if (globalLiveStream) {
          globalLiveStream.close();
          globalLiveStream = null;
        }

        currentConnectedSessionId = sessionId;
        set({ liveSessionId: sessionId, liveConnected: false });

        globalLiveStream = openLiveStream(sessionId, {
          onOpen: () => set({ liveConnected: true }),
          onClose: () => set({ liveConnected: false }),
          onStatus: (st) => set({ liveStatus: st }),
          onForecast: (f) => {
            if (get().livePaused) return;
            set((st) => ({
              liveForecasts: [...st.liveForecasts.slice(-199), f],
            }));
          },
          onError: (err) => {
            console.warn("[SENTINEL live stream error]", err);
          },
        });
      },

      stopLiveSession: async () => {
        const id = get().liveSessionId;
        if (globalLiveStream) {
          globalLiveStream.close();
          globalLiveStream = null;
        }
        currentConnectedSessionId = null;

        if (id) {
          try {
            await api.deleteLiveSession(id);
          } catch {
            /* ignore deletion error */
          }
        }

        set({
          liveSessionId: null,
          liveStatus: null,
          liveConnected: false,
          livePaused: false,
        });
      },

      toggleLivePause: () => set((s) => ({ livePaused: !s.livePaused })),

      pushLiveForecast: (f) => {
        if (get().livePaused) return;
        set((st) => ({
          liveForecasts: [...st.liveForecasts.slice(-199), f],
        }));
      },

      resetLive: () => {
        if (globalLiveStream) {
          globalLiveStream.close();
          globalLiveStream = null;
        }
        currentConnectedSessionId = null;
        set({
          liveSessionId: null,
          liveStatus: null,
          liveForecasts: [],
          liveConnected: false,
          livePaused: false,
        });
      },

      reconnectLiveIfActive: async () => {
        const id = get().liveSessionId;
        if (!id) return;
        if (globalLiveStream && currentConnectedSessionId === id) return;

        try {
          const session = await api.getLiveSession(id);
          if (session && session.state !== "stopped" && session.state !== "error") {
            set({ liveStatus: session });
            get().attachLiveStream(id);
          } else {
            get().resetLive();
          }
        } catch {
          // Session likely expired on server restart or cleanup
          get().resetLive();
        }
      },
    }),
    {
      name: "sentinel-wm-store",
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? window.localStorage : (null as any)
      ),
      partialize: (s) => ({
        activeSource: s.activeSource,
        result: s.result,
        selectedAnchor: s.selectedAnchor,
        liveMode: s.liveMode,
        scenarioForm: s.scenarioForm,
        captureIface: s.captureIface,
        captureBpf: s.captureBpf,
        liveSessionId: s.liveSessionId,
        liveStatus: s.liveStatus,
        liveForecasts: s.liveForecasts.slice(-200),
        livePaused: s.livePaused,
        csv: s.csv
          ? {
              name: s.csv.name,
              header: s.csv.header,
              rows: s.csv.rows.slice(0, 50),
              summary: s.csv.summary,
              match: s.csv.match,
            }
          : null,
      }),
    }
  )
);
