"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronRight, Sliders } from "lucide-react";
import { ChatResponse } from "@/lib/types";

interface TechnicalDetailsProps {
  data: Partial<ChatResponse>;
}

export const TechnicalDetails: React.FC<TechnicalDetailsProps> = ({ data }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="pt-1">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 text-xs text-foreground-muted hover:text-foreground-secondary py-1 transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="w-3.5 h-3.5" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5" />
        )}
        <Sliders className="w-3 h-3" />
        <span>Technical details</span>
      </button>

      {isOpen && (
        <div className="mt-2 p-3 rounded bg-surface border border-border text-xs space-y-3">
          {/* Retrieval telemetry */}
          <div>
            <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5">
              Retrieval & Routing
            </div>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
              <div className="p-2 rounded bg-background border border-border">
                <span className="text-foreground-muted block">Strategy:</span>
                <span className="text-foreground">{data.strategy || "Hybrid RRF"}</span>
              </div>
              <div className="p-2 rounded bg-background border border-border">
                <span className="text-foreground-muted block">Query Class:</span>
                <span className="text-foreground capitalize">{data.query_type || "General"}</span>
              </div>
              <div className="p-2 rounded bg-background border border-border">
                <span className="text-foreground-muted block">Generator Persona:</span>
                <span className="text-foreground capitalize">{data.role_used || "Analyst"}</span>
              </div>
              <div className="p-2 rounded bg-background border border-border">
                <span className="text-foreground-muted block">Graph RAG:</span>
                <span className="text-foreground">{data.graph_rag_used ? "Active" : "Bypassed"}</span>
              </div>
              <div className="p-2 rounded bg-background border border-border">
                <span className="text-foreground-muted block">Web Research:</span>
                <span className="text-foreground">{data.web_research_used ? "Active" : "Bypassed"}</span>
              </div>
            </div>
          </div>

          {/* Critic & Consensus evaluation */}
          {data.critique && Object.keys(data.critique).length > 0 && (
            <div>
              <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5">
                Evaluation Metrics
              </div>
              <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                {data.consensus_score !== undefined && (
                  <div className="p-2 rounded bg-background border border-border">
                    <span className="text-foreground-muted block">Reviewer Consensus:</span>
                    <span className="text-foreground">{data.consensus_score.toFixed(1)} / 10.0</span>
                  </div>
                )}
                {data.iterations !== undefined && (
                  <div className="p-2 rounded bg-background border border-border">
                    <span className="text-foreground-muted block">Iterations:</span>
                    <span className="text-foreground">{data.iterations}</span>
                  </div>
                )}
                {data.critique.faithfulness !== undefined && (
                  <div className="p-2 rounded bg-background border border-border">
                    <span className="text-foreground-muted block">Faithfulness:</span>
                    <span className="text-foreground">{data.critique.faithfulness} / 10</span>
                  </div>
                )}
                {data.critique.completeness !== undefined && (
                  <div className="p-2 rounded bg-background border border-border">
                    <span className="text-foreground-muted block">Completeness:</span>
                    <span className="text-foreground">{data.critique.completeness} / 10</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Identifiers */}
          {data.run_id && (
            <div className="flex items-center justify-between text-[10px] text-foreground-muted font-mono pt-1 border-t border-border">
              <span>Run ID: {data.run_id}</span>
              {data.session_id && <span>Session: {data.session_id}</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
