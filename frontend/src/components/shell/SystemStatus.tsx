"use client";

import React, { useEffect, useState } from "react";
import { CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { HealthResponse } from "@/lib/types";
import { Modal } from "@/components/ui/Modal";

export const SystemStatus: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const fetchHealth = async () => {
    try {
      setLoading(true);
      const data = await api.getHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const isHealthy = health?.status === "healthy";

  return (
    <>
      <button
        onClick={() => setIsModalOpen(true)}
        className="w-full flex items-center justify-between px-2.5 py-1.5 rounded text-xs text-foreground-secondary hover:text-foreground hover:bg-surface-hover transition-colors"
        title="View system diagnostics"
      >
        <span className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isHealthy ? "bg-success" : "bg-warning"
            }`}
          />
          <span>{isHealthy ? "Local system ready" : "System initializing"}</span>
        </span>
      </button>

      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Local System Status"
        description="Live status of underlying LocalMind-RAG backend modules."
      >
        <div className="space-y-3">
          <div className="flex items-center justify-between p-2.5 rounded bg-background border border-border text-xs">
            <span className="text-foreground-secondary">API Gateway</span>
            <span className="flex items-center gap-1.5 text-success font-mono">
              <CheckCircle2 className="w-3.5 h-3.5" /> Healthy
            </span>
          </div>

          <div className="flex items-center justify-between p-2.5 rounded bg-background border border-border text-xs">
            <span className="text-foreground-secondary">Vector Store (ChromaDB)</span>
            <span className="flex items-center gap-1.5 font-mono text-foreground">
              {health?.vector_store?.status === "healthy" ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                  <span>{health.vector_store.collection_count} vectors</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-warning" />
                  <span>Offline</span>
                </>
              )}
            </span>
          </div>

          <div className="flex items-center justify-between p-2.5 rounded bg-background border border-border text-xs">
            <span className="text-foreground-secondary">Local LLM / Ollama</span>
            <span className="flex items-center gap-1.5 font-mono text-foreground">
              {health?.llm?.status === "healthy" ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                  <span>{health.llm.model_name || "Ready"}</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-warning" />
                  <span>{health?.llm?.status || "Unavailable"}</span>
                </>
              )}
            </span>
          </div>

          <div className="flex items-center justify-between p-2.5 rounded bg-background border border-border text-xs">
            <span className="text-foreground-secondary">Knowledge Graph Store</span>
            <span className="flex items-center gap-1.5 font-mono text-foreground">
              {health?.graph_store?.status === "healthy" ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-success" />
                  <span>
                    {health.graph_store.node_count} nodes · {health.graph_store.edge_count} edges
                  </span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-warning" />
                  <span>Uninitialized</span>
                </>
              )}
            </span>
          </div>

          <div className="pt-2 flex justify-end">
            <button
              onClick={fetchHealth}
              disabled={loading}
              className="flex items-center gap-1.5 text-xs text-foreground-secondary hover:text-foreground p-1.5 rounded transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>Refresh Status</span>
            </button>
          </div>
        </div>
      </Modal>
    </>
  );
};
