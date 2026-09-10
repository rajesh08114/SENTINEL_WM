import { create } from "zustand";
import type { AnchorForecast, ForecastResponse, LiveSessionInfo } from "./types";
import type { ColumnMatch } from "./schema";
import type { CsvSummary } from "./csv";

export interface CsvState {
  file: File;
  name: string;
  header: string[];
  rows: string[][];
  summary: CsvSummary;
  match: ColumnMatch;
}

interface Store {
  // uploaded-CSV forecast
  csv: CsvState | null;
  result: ForecastResponse | null;
  selectedAnchor: number;
  setCsv: (c: CsvState | null) => void;
  setResult: (r: ForecastResponse | null) => void;
  selectAnchor: (i: number) => void;

  // live session
  liveSessionId: string | null;
  liveStatus: LiveSessionInfo | null;
  liveForecasts: AnchorForecast[];
  setLiveSession: (id: string | null) => void;
  setLiveStatus: (s: LiveSessionInfo | null) => void;
  pushLiveForecast: (f: AnchorForecast) => void;
  resetLive: () => void;
}

export const useStore = create<Store>((set) => ({
  csv: null,
  result: null,
  selectedAnchor: 0,
  setCsv: (c) => set({ csv: c, result: null }),
  setResult: (r) => set({ result: r, selectedAnchor: 0 }),
  selectAnchor: (i) => set({ selectedAnchor: i }),

  liveSessionId: null,
  liveStatus: null,
  liveForecasts: [],
  setLiveSession: (id) => set({ liveSessionId: id }),
  setLiveStatus: (s) => set({ liveStatus: s }),
  pushLiveForecast: (f) =>
    set((st) => ({ liveForecasts: [...st.liveForecasts.slice(-199), f] })),
  resetLive: () => set({ liveSessionId: null, liveStatus: null, liveForecasts: [] }),
}));
