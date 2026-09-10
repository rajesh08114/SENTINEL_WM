import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { openLiveStream } from "../ws";

class MockWS {
  static instances: MockWS[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];
  closed = false;
  constructor(url: string) {
    this.url = url;
    MockWS.instances.push(this);
  }
  send(s: string) {
    this.sent.push(s);
  }
  close() {
    this.closed = true;
    this.onclose?.();
  }
  emit(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
}

beforeEach(() => {
  MockWS.instances = [];
  vi.stubGlobal("WebSocket", MockWS as unknown as typeof WebSocket);
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("openLiveStream", () => {
  it("delivers forecast + status frames", () => {
    const forecasts: unknown[] = [];
    const statuses: unknown[] = [];
    openLiveStream("s1", {
      onForecast: (f) => forecasts.push(f),
      onStatus: (s) => statuses.push(s),
    });
    const ws = MockWS.instances[0];
    ws.onopen?.();
    ws.emit({ type: "status", id: "s1", state: "running" });
    ws.emit({ type: "forecast", alert: true, horizon: [] });
    expect(statuses).toHaveLength(1);
    expect(forecasts).toHaveLength(1);
  });

  it("reconnects after an unexpected close, with backoff", () => {
    openLiveStream("s1", {});
    expect(MockWS.instances).toHaveLength(1);
    MockWS.instances[0].close(); // unexpected (no bye)
    vi.advanceTimersByTime(1000);
    expect(MockWS.instances).toHaveLength(2);
  });

  it("stops reconnecting after close()", () => {
    const stream = openLiveStream("s1", {});
    stream.close();
    MockWS.instances[0].onclose?.();
    vi.advanceTimersByTime(30_000);
    expect(MockWS.instances).toHaveLength(1);
  });

  it("stops reconnecting after a bye frame", () => {
    openLiveStream("s1", {});
    MockWS.instances[0].emit({ type: "bye" });
    MockWS.instances[0].onclose?.();
    vi.advanceTimersByTime(30_000);
    expect(MockWS.instances).toHaveLength(1);
  });
});
