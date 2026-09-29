"use client";

import React, { useState } from "react";
import { Check, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { ExecutionEvent, ExecutionStage } from "@/lib/types";

interface ExecutionTraceProps {
  events: ExecutionEvent[];
  isStreaming: boolean;
  summaryText?: string;
}

const STAGE_LABELS: Record<ExecutionStage, string> = {
  connected: "Handshake established",
  firewall: "Input safety validation",
  memory: "Context retrieval",
  supervisor: "Query analysis & routing",
  planner: "Research decomposition",
  retrieval: "Hybrid evidence retrieval",
  web_research: "External web research",
  generation: "Answer generation",
  critique: "Faithfulness & completeness verification",
  review: "Consensus evaluation",
  refinement: "Refinement loop",
  tsl: "Trust & safety inspection",
  final: "Execution finalized",
  error: "Execution encountered error",
};

export const ExecutionTrace: React.FC<ExecutionTraceProps> = ({
  events,
  isStreaming,
  summaryText,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (events.length === 0 && !isStreaming) {
    return null;
  }

  // During streaming, display the active trace
  if (isStreaming) {
    const latestEvent = events[events.length - 1]?.event || "supervisor";

    return (
      <div className="my-2.5 p-3 rounded bg-surface border border-border text-xs">
        <div className="flex items-center gap-2 text-foreground font-medium mb-2">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
          <span>Researching your knowledge</span>
        </div>

        <div className="space-y-1 pl-1">
          {events.map((evt, idx) => {
            const isLast = idx === events.length - 1;
            const label = STAGE_LABELS[evt.event] || evt.event;

            return (
              <div
                key={`${evt.event}-${idx}`}
                className="flex items-center gap-2 text-foreground-secondary text-[11px]"
              >
                {isLast ? (
                  <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                ) : (
                  <Check className="w-3 h-3 text-success shrink-0" />
                )}
                <span className={isLast ? "text-foreground font-medium" : ""}>
                  {label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Once completed, collapse into a quiet progressive-disclosure row
  return (
    <div className="my-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 py-1 px-2 rounded hover:bg-surface text-xs text-foreground-secondary hover:text-foreground transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-foreground-muted" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-foreground-muted" />
        )}
        <span className="font-mono text-[11px]">{summaryText || "Execution trace"}</span>
      </button>

      {isExpanded && (
        <div className="mt-1.5 ml-4 pl-3 border-l border-border space-y-1 text-[11px] text-foreground-secondary">
          {events.map((evt, idx) => (
            <div key={`${evt.event}-${idx}`} className="flex items-center gap-2">
              <Check className="w-3 h-3 text-success shrink-0" />
              <span>{STAGE_LABELS[evt.event] || evt.event}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
