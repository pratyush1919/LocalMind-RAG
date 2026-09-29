"use client";

import React from "react";
import { CitationBadge } from "./CitationBadge";
import { ExecutionTrace } from "./ExecutionTrace";
import { TechnicalDetails } from "./TechnicalDetails";
import { ChatResponse, ExecutionEvent } from "@/lib/types";
import { SourceEvidence } from "./SourceInspector";
import { Sparkles, ShieldAlert } from "lucide-react";

export interface ChatMessageUI {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  consensus_score?: number | null;
  sources?: string[];
  executionEvents?: ExecutionEvent[];
  isStreaming?: boolean;
  metadata?: Partial<ChatResponse>;
  guardrail_blocked?: boolean;
  guardrail_reason?: string | null;
  usedMemory?: boolean;
}

interface MessageItemProps {
  message: ChatMessageUI;
  onSelectCitation: (source: SourceEvidence) => void;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, onSelectCitation }) => {
  if (message.role === "user") {
    return (
      <div className="py-3 px-1">
        <h2 className="text-base sm:text-lg font-medium text-foreground tracking-tight leading-snug">
          {message.content}
        </h2>
      </div>
    );
  }

  // Parse citations in assistant text (e.g. "[Source 1]", "[1]")
  const renderContentWithCitations = (text: string) => {
    // Regex matching [Source X] or [X]
    const regex = /(\[(?:Source\s+)?\d+\])/gi;
    const parts = text.split(regex);

    return parts.map((part, index) => {
      if (regex.test(part)) {
        // Find matching source if available
        const sourceIndexMatch = part.match(/\d+/);
        const sourceIdx = sourceIndexMatch ? parseInt(sourceIndexMatch[0], 10) - 1 : 0;
        const matchedDetail = message.metadata?.source_details?.[sourceIdx];
        const filename = matchedDetail?.filename || message.sources?.[sourceIdx] || message.sources?.[0] || `Source Document`;

        const evidence: SourceEvidence = matchedDetail || {
          filename,
          file_type: filename.split(".").pop() || "txt",
          doc_category: message.metadata?.query_type || "general",
          strategy: message.metadata?.strategy || "Hybrid RAG",
          source_type: "document",
        };

        return (
          <CitationBadge
            key={`cit-${index}`}
            label={part}
            onClick={() => onSelectCitation(evidence)}
          />
        );
      }
      return <span key={`text-${index}`}>{part}</span>;
    });
  };

  // Guardrail blocked message
  if (message.guardrail_blocked) {
    return (
      <div className="my-2 p-3.5 rounded bg-surface border border-border text-xs space-y-1.5">
        <div className="flex items-center gap-2 text-warning font-medium">
          <ShieldAlert className="w-4 h-4 text-warning" />
          <span>Request filtered by Trust & Safety policy</span>
        </div>
        <div className="text-foreground-secondary text-[11px] leading-relaxed">
          {message.guardrail_reason || "This query violates automated safety or privacy boundaries."}
        </div>
      </div>
    );
  }

  const retrievalDesc = message.metadata?.web_research_used
    ? "Local + Web research"
    : message.metadata?.strategy || "Hybrid retrieval";

  // Summary text for collapsed execution trace
  const summaryText =
    message.sources && message.sources.length > 0
      ? `${message.sources.length} source${message.sources.length > 1 ? "s" : ""} · ${retrievalDesc}${
          message.consensus_score
            ? ` · Verified ${message.consensus_score.toFixed(1)}/10`
            : ""
        }`
      : "Execution trace";

  return (
    <div className="py-3 px-1 space-y-2 border-b border-border/50 last:border-b-0">
      {/* Execution Trace (real SSE events) */}
      {(message.isStreaming || (message.executionEvents && message.executionEvents.length > 0)) && (
        <ExecutionTrace
          events={message.executionEvents || []}
          isStreaming={!!message.isStreaming}
          summaryText={summaryText}
        />
      )}

      {/* Answer Body */}
      {message.content ? (
        <div className="text-sm sm:text-base leading-relaxed text-foreground font-normal whitespace-pre-wrap">
          {renderContentWithCitations(message.content)}
        </div>
      ) : message.isStreaming ? (
        <div className="text-xs text-foreground-muted italic">
          Synthesizing verified answer from retrieved evidence...
        </div>
      ) : null}

      {/* Subtle Memory Indicator */}
      {message.usedMemory && (
        <div className="text-[11px] text-foreground-muted flex items-center gap-1.5 pt-1">
          <span>↳ Conversation context used</span>
        </div>
      )}

      {/* Technical Details Accordion */}
      {message.metadata && !message.isStreaming && (
        <TechnicalDetails data={message.metadata} />
      )}
    </div>
  );
};
