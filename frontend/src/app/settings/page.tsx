import React from "react";
import { Settings as SettingsIcon, Server, Database, Cpu, ShieldCheck } from "lucide-react";

export default function SettingsPage() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  return (
    <div className="flex-1 h-full overflow-y-auto px-6 py-6 max-w-4xl mx-auto w-full space-y-6">
      <div className="border-b border-border pb-4">
        <h1 className="text-xl sm:text-2xl font-semibold text-foreground tracking-tight flex items-center gap-2">
          <SettingsIcon className="w-5 h-5 text-foreground-secondary" />
          <span>Settings & Configuration</span>
        </h1>
        <p className="text-xs text-foreground-secondary mt-1">
          Local configuration parameters and system environment.
        </p>
      </div>

      <div className="space-y-4">
        {/* Backend API Configuration */}
        <div className="p-4 rounded-lg bg-surface border border-border space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground">
            <Server className="w-4 h-4 text-accent" />
            <span>FastAPI Backend Gateway</span>
          </div>
          <p className="text-xs text-foreground-secondary">
            The REST and SSE streaming endpoint used by this frontend client.
          </p>
          <div className="p-2 rounded bg-background border border-border font-mono text-xs text-foreground select-all">
            {apiUrl}
          </div>
        </div>

        {/* Vector Store Configuration */}
        <div className="p-4 rounded-lg bg-surface border border-border space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground">
            <Database className="w-4 h-4 text-accent" />
            <span>Vector & Sparse Indexes</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
            <div className="p-2.5 rounded bg-background border border-border">
              <span className="text-foreground-muted block text-[11px]">Dense Store:</span>
              <span className="text-foreground">ChromaDB (./vectorstore)</span>
            </div>
            <div className="p-2.5 rounded bg-background border border-border">
              <span className="text-foreground-muted block text-[11px]">Sparse Index:</span>
              <span className="text-foreground">BM25 (./vectorstore/bm25)</span>
            </div>
          </div>
        </div>

        {/* Local Model Factory */}
        <div className="p-4 rounded-lg bg-surface border border-border space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground">
            <Cpu className="w-4 h-4 text-accent" />
            <span>Inference & Embedding Models</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
            <div className="p-2.5 rounded bg-background border border-border">
              <span className="text-foreground-muted block text-[11px]">Embedding Model:</span>
              <span className="text-foreground">BAAI/bge-small-en-v1.5</span>
            </div>
            <div className="p-2.5 rounded bg-background border border-border">
              <span className="text-foreground-muted block text-[11px]">Local LLM:</span>
              <span className="text-foreground">Ollama (mistral-rag)</span>
            </div>
          </div>
        </div>

        {/* Safety & Guardrails */}
        <div className="p-4 rounded-lg bg-surface border border-border space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground">
            <ShieldCheck className="w-4 h-4 text-accent" />
            <span>Trust & Safety Layer</span>
          </div>
          <p className="text-xs text-foreground-secondary leading-relaxed">
            Multi-tier input firewall (regex injection patterns + LLM jailbreak scanner) and output TSL scanning (PII, toxicity, risk evaluation) run on all queries.
          </p>
        </div>
      </div>
    </div>
  );
}
