/**
 * frontend/src/lib/api.ts
 * Centralized HTTP API client for LocalMind-RAG FastAPI backend.
 */

import {
  ChatRequest,
  ChatResponse,
  DocumentListResponse,
  DocumentUploadResponse,
  DocumentDeleteResponse,
  GraphData,
  HealthResponse,
  SessionDeleteResponse,
  SessionDetail,
  SessionListResponse,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const headers = {
    Accept: "application/json",
    ...options.headers,
  };

  const response = await fetch(url, { ...options, headers });

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Use fallback errorDetail
    }
    throw new Error(errorDetail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  // System Health
  getHealth: (): Promise<HealthResponse> => request<HealthResponse>("/api/health"),

  // Chat API
  chatSync: (body: ChatRequest): Promise<ChatResponse> =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  // Session Management
  listSessions: (): Promise<SessionListResponse> => request<SessionListResponse>("/api/chat/sessions"),

  getSessionHistory: (sessionId: string): Promise<SessionDetail> =>
    request<SessionDetail>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`),

  deleteSession: (sessionId: string): Promise<SessionDeleteResponse> =>
    request<SessionDeleteResponse>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }),

  // Document Management
  listDocuments: (): Promise<DocumentListResponse> => request<DocumentListResponse>("/api/documents"),

  uploadDocument: (formData: FormData): Promise<DocumentUploadResponse> =>
    request<DocumentUploadResponse>("/api/documents/upload", {
      method: "POST",
      body: formData,
    }),

  deleteDocument: (filename: string): Promise<DocumentDeleteResponse> =>
    request<DocumentDeleteResponse>(`/api/documents/${encodeURIComponent(filename)}`, {
      method: "DELETE",
    }),

  // Knowledge Graph
  getGraphData: (): Promise<GraphData> => request<GraphData>("/api/graph"),
};
