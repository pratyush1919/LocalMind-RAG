/**
 * frontend/src/lib/types.ts
 * Strict TypeScript models matching the LocalMind-RAG FastAPI backend contracts.
 */

// ============================================================================
// System Health
// ============================================================================

export interface VectorStoreHealth {
  status: string;
  collection_count: number;
}

export interface LLMHealth {
  status: string;
  model_name: string;
}

export interface GraphStoreHealth {
  status: string;
  node_count: number;
  edge_count: number;
}

export interface HealthResponse {
  status: string;
  vector_store: VectorStoreHealth;
  llm: LLMHealth;
  graph_store: GraphStoreHealth;
  timestamp: string;
}

// ============================================================================
// Chat API & SSE Models
// ============================================================================

export interface SourceEvidence {
  filename: string;
  title?: string;
  url?: string;
  domain?: string;
  source_type?: "document" | "web" | "graph";
  file_type?: string;
  page_number?: number;
  section?: string;
  snippet?: string;
  doc_category?: string;
  strategy?: string;
}

export interface ChatRequest {
  query: string;
  session_id?: string | null;
  user_role?: "analyst" | "researcher" | "developer" | "general";
  use_memory?: boolean;
  use_web_research?: boolean;
}

export interface CritiqueSummary {
  faithfulness?: number;
  completeness?: number;
  clarity?: number;
  adherence?: number;
}

export interface TslSummary {
  risk_score?: number;
  passed?: boolean;
  pii_found?: boolean;
}

export interface ChatResponse {
  answer: string;
  consensus_score: number;
  is_final: boolean;
  iterations: number;
  query_type: string;
  role_used: string;
  citations: string[];
  sources: string[];
  source_details?: SourceEvidence[];
  strategy: string;
  graph_rag_used: boolean;
  web_research_used?: boolean;
  critique: CritiqueSummary;
  run_id: string;
  session_id: string;
  user_role: string;
  tsl?: TslSummary;
  guardrail_blocked: boolean;
  guardrail_reason?: string | null;
}

export type ExecutionStage =
  | "connected"
  | "firewall"
  | "memory"
  | "supervisor"
  | "planner"
  | "retrieval"
  | "web_research"
  | "generation"
  | "critique"
  | "review"
  | "refinement"
  | "tsl"
  | "final"
  | "error";

export interface ExecutionEvent {
  event: ExecutionStage;
  data: Record<string, unknown>;
  timestamp: number;
}

// ============================================================================
// Session Models
// ============================================================================

export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionListResponse {
  sessions: SessionSummary[];
  total: number;
}

export interface SessionMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  citations?: string[];
  consensus_score?: number | null;
}

export interface SessionDetail {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
}

export interface SessionDeleteResponse {
  success: boolean;
  session_id: string;
  message: string;
}

// ============================================================================
// Document Models
// ============================================================================

export interface DocumentItem {
  filename: string;
  file_type: string;
  doc_category: string;
  chunk_count: number;
  created_at?: string | null;
}

export interface DocumentListResponse {
  total_documents: number;
  total_chunks: number;
  documents: DocumentItem[];
}

export interface DocumentUploadResponse {
  status: "success" | "error";
  filename: string;
  strategy: string;
  doc_category: string;
  chunks_created: number;
  total_chunks_in_collection: number;
  graph_built: boolean;
  elapsed_seconds: number;
  message?: string | null;
}

export interface DocumentDeleteResponse {
  success: boolean;
  filename: string;
  chunks_removed: number;
  message: string;
}

// ============================================================================
// Knowledge Graph Models
// ============================================================================

export interface GraphNode {
  id: string;
  label: string;
  frequency?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  predicate: string;
  source_file?: string;
  chunk_id?: string;
  weight?: number;
}

export interface GraphData {
  total_nodes: number;
  total_edges: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}
