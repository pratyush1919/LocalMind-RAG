/**
 * frontend/src/lib/sse.ts
 * Robust SSE client for POST /api/chat/stream with line-buffered event parsing.
 */

import { ChatRequest, ChatResponse, ExecutionStage } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export interface StreamCallbacks {
  onEvent: (event: ExecutionStage, data: Record<string, unknown>) => void;
  onFinal: (response: ChatResponse) => void;
  onError: (error: Error) => void;
}

export function streamChatQuery(
  request: ChatRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): () => void {
  const controller = new AbortController();
  const internalSignal = signal || controller.signal;

  (async () => {
    try {
      const response = await fetch(`${BASE_URL}/api/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(request),
        signal: internalSignal,
      });

      if (!response.ok) {
        let errMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData.detail) {
            errMessage = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch {
          // fallback
        }
        throw new Error(errMessage);
      }

      if (!response.body) {
        throw new Error("Response body is null");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Keep trailing incomplete line in buffer
        buffer = lines.pop() || "";

        let currentEvent: ExecutionStage = "supervisor";
        let currentData = "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) {
            // Empty line indicates event boundary
            if (currentData) {
              try {
                const parsedData = JSON.parse(currentData);
                if (currentEvent === "final") {
                  callbacks.onFinal(parsedData as ChatResponse);
                } else {
                  callbacks.onEvent(currentEvent, parsedData);
                }
              } catch (e) {
                console.warn("[SSE] Could not parse event JSON:", currentData, e);
              }
              currentData = "";
            }
            continue;
          }

          if (trimmed.startsWith("event:")) {
            currentEvent = trimmed.slice(6).trim() as ExecutionStage;
          } else if (trimmed.startsWith("data:")) {
            currentData = trimmed.slice(5).trim();
          }
        }
      }

      // Handle any remainder
      if (buffer.trim()) {
        const trimmed = buffer.trim();
        if (trimmed.startsWith("data:")) {
          try {
            const parsedData = JSON.parse(trimmed.slice(5).trim());
            callbacks.onFinal(parsedData as ChatResponse);
          } catch {
            // ignore remainder
          }
        }
      }
    } catch (err: unknown) {
      if ((err as Error).name === "AbortError") {
        return; // Normal cancellation
      }
      callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    }
  })();

  return () => controller.abort();
}
