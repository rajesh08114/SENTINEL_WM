import { wsBase } from "./api";
import type { AnchorForecast, LiveFrame, LiveSessionInfo } from "./types";

export interface LiveStreamHandlers {
  onForecast?: (f: AnchorForecast) => void;
  onStatus?: (s: LiveSessionInfo) => void;
  onError?: (detail: string) => void;
  onOpen?: () => void;
  onClose?: (bye: boolean) => void;
}

export interface LiveStream {
  close: () => void;
}

/** Subscribe to /live/sessions/{id}/stream with reconnect + backoff.
 *  Stops reconnecting once a {type:"bye"} frame is seen or close() is called. */
export function openLiveStream(sessionId: string, h: LiveStreamHandlers): LiveStream {
  let ws: WebSocket | null = null;
  let closed = false;
  let gotBye = false;
  let backoff = 1000;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const connect = () => {
    if (closed) return;
    try {
      ws = new WebSocket(`${wsBase()}/live/sessions/${sessionId}/stream`);
    } catch {
      schedule();
      return;
    }
    ws.onopen = () => {
      backoff = 1000;
      h.onOpen?.();
    };
    ws.onmessage = (ev) => {
      let msg: LiveFrame;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "forecast") h.onForecast?.(msg as unknown as AnchorForecast);
      else if (msg.type === "status") h.onStatus?.(msg as unknown as LiveSessionInfo);
      else if (msg.type === "error") h.onError?.(msg.detail);
      else if (msg.type === "bye") {
        gotBye = true;
        try {
          ws?.close();
        } catch {
          /* ignore */
        }
      }
    };
    ws.onclose = () => {
      h.onClose?.(gotBye);
      if (!closed && !gotBye) schedule();
    };
    ws.onerror = () => {
      try {
        ws?.close();
      } catch {
        /* ignore */
      }
    };
  };

  const schedule = () => {
    if (closed || gotBye) return;
    timer = setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 15000);
  };

  connect();

  return {
    close() {
      closed = true;
      if (timer) clearTimeout(timer);
      try {
        ws?.send(JSON.stringify({ type: "stop" }));
      } catch {
        /* ignore */
      }
      try {
        ws?.close();
      } catch {
        /* ignore */
      }
    },
  };
}
