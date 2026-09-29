"use client";

import React from "react";
import { Network, ArrowRight, FileText, ExternalLink } from "lucide-react";
import { GraphEdge, GraphNode } from "@/lib/types";
import { Sheet } from "@/components/ui/Sheet";
import { Button } from "@/components/ui/Button";
import Link from "next/link";

interface NodeInspectorProps {
  node: GraphNode | null;
  edges: GraphEdge[];
  isOpen: boolean;
  onClose: () => void;
  onSelectNode: (nodeId: string) => void;
}

export const NodeInspector: React.FC<NodeInspectorProps> = ({
  node,
  edges,
  isOpen,
  onClose,
  onSelectNode,
}) => {
  if (!node) return null;

  // Find direct relationships
  const outgoing = edges.filter(
    (e) => (typeof e.source === "object" ? (e.source as { id: string }).id : e.source) === node.id
  );
  const incoming = edges.filter(
    (e) => (typeof e.target === "object" ? (e.target as { id: string }).id : e.target) === node.id
  );

  // Extract source provenance documents
  const sourceDocs = Array.from(
    new Set(
      edges
        .filter(
          (e) =>
            (typeof e.source === "object" ? (e.source as { id: string }).id : e.source) === node.id ||
            (typeof e.target === "object" ? (e.target as { id: string }).id : e.target) === node.id
        )
        .map((e) => e.source_file)
        .filter(Boolean) as string[]
    )
  );

  return (
    <Sheet
      isOpen={isOpen}
      onClose={onClose}
      title={node.label}
      subtitle="Entity Concept · Knowledge Graph"
    >
      <div className="space-y-4 text-xs">
        {/* Entity Card */}
        <div className="p-3 rounded bg-background border border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Network className="w-4 h-4 text-accent" />
            <div>
              <div className="font-medium text-foreground">{node.label}</div>
              <div className="text-[11px] text-foreground-muted font-mono">id: {node.id}</div>
            </div>
          </div>
          <span className="font-mono text-foreground-secondary bg-surface px-2 py-1 rounded border border-border">
            {node.frequency || 1} reference{(node.frequency || 1) !== 1 ? "s" : ""}
          </span>
        </div>

        {/* Outgoing Relationships */}
        {outgoing.length > 0 && (
          <div>
            <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5">
              Outgoing Relationships ({outgoing.length})
            </div>
            <div className="divide-y divide-border border border-border rounded bg-background">
              {outgoing.map((e, idx) => {
                const targetId =
                  typeof e.target === "object" ? (e.target as { id: string }).id : e.target;
                return (
                  <button
                    key={`out-${idx}`}
                    onClick={() => onSelectNode(targetId)}
                    className="w-full text-left p-2.5 hover:bg-surface-hover transition-colors flex items-center justify-between text-xs"
                  >
                    <span className="font-mono text-foreground-muted italic">
                      — {e.predicate} →
                    </span>
                    <span className="font-medium text-foreground flex items-center gap-1">
                      <span>{targetId}</span>
                      <ArrowRight className="w-3 h-3 text-foreground-muted" />
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Incoming Relationships */}
        {incoming.length > 0 && (
          <div>
            <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5">
              Incoming Relationships ({incoming.length})
            </div>
            <div className="divide-y divide-border border border-border rounded bg-background">
              {incoming.map((e, idx) => {
                const sourceId =
                  typeof e.source === "object" ? (e.source as { id: string }).id : e.source;
                return (
                  <button
                    key={`in-${idx}`}
                    onClick={() => onSelectNode(sourceId)}
                    className="w-full text-left p-2.5 hover:bg-surface-hover transition-colors flex items-center justify-between text-xs"
                  >
                    <span className="font-medium text-foreground">{sourceId}</span>
                    <span className="font-mono text-foreground-muted italic">
                      — {e.predicate} →
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Source Documents Provenance */}
        <div>
          <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5">
            Source Provenance
          </div>
          {sourceDocs.length > 0 ? (
            <div className="space-y-1">
              {sourceDocs.map((doc, idx) => (
                <div
                  key={`src-${idx}`}
                  className="flex items-center gap-2 p-2 rounded bg-background border border-border text-xs text-foreground font-mono"
                >
                  <FileText className="w-3.5 h-3.5 text-foreground-muted" />
                  <span className="truncate">{doc}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-2.5 rounded bg-background border border-border text-foreground-muted italic">
              Extracted from indexed corpus chunks.
            </div>
          )}
        </div>

        {/* Primary Action Button: Explore evidence → */}
        <div className="pt-2">
          <Link href="/documents">
            <Button variant="primary" size="sm" className="w-full justify-center gap-2">
              <span>Explore evidence</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </Button>
          </Link>
        </div>
      </div>
    </Sheet>
  );
};
