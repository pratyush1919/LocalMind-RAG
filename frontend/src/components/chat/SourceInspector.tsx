"use client";

import React from "react";
import { FileText, Globe, ExternalLink, Bookmark } from "lucide-react";
import { Sheet } from "@/components/ui/Sheet";
import { Button } from "@/components/ui/Button";
import Link from "next/link";
import { SourceEvidence } from "@/lib/types";

export type { SourceEvidence };

interface SourceInspectorProps {
  source: SourceEvidence | null;
  isOpen: boolean;
  onClose: () => void;
}

export const SourceInspector: React.FC<SourceInspectorProps> = ({
  source,
  isOpen,
  onClose,
}) => {
  if (!source) return null;

  const isWebSource = source.source_type === "web" || !!source.url;
  const subtitle = isWebSource
    ? (source.domain ? `WEB · ${source.domain}` : "WEB SOURCE")
    : `${source.file_type ? source.file_type.toUpperCase() : "DOCUMENT"} · Local knowledge`;

  return (
    <Sheet
      isOpen={isOpen}
      onClose={onClose}
      title={source.title || source.filename}
      subtitle={subtitle}
    >
      <div className="space-y-4 text-xs">
        {/* Source Header Info */}
        <div className="p-3 rounded bg-background border border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isWebSource ? (
              <Globe className="w-4 h-4 text-accent shrink-0" />
            ) : (
              <FileText className="w-4 h-4 text-foreground-secondary shrink-0" />
            )}
            <div>
              <div className="font-medium text-foreground">{source.title || source.filename}</div>
              {isWebSource ? (
                source.domain && (
                  <div className="text-[11px] text-foreground-muted font-mono">
                    {source.domain}
                  </div>
                )
              ) : (
                source.doc_category && (
                  <div className="text-[11px] text-foreground-muted capitalize">
                    Category: {source.doc_category}
                  </div>
                )
              )}
            </div>
          </div>
          {!isWebSource && source.page_number && (
            <span className="font-mono text-foreground-secondary bg-surface px-2 py-1 rounded border border-border">
              Page {source.page_number}
            </span>
          )}
        </div>

        {/* Relevant Evidence Passage */}
        {source.snippet ? (
          <div>
            <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
              <Bookmark className="w-3.5 h-3.5" />
              <span>Retrieved Evidence</span>
            </div>
            <div className="p-3 rounded bg-background border border-border font-mono text-[11px] leading-relaxed text-foreground-secondary whitespace-pre-wrap">
              &ldquo;{source.snippet}&rdquo;
            </div>
          </div>
        ) : (
          <div className="p-3 rounded bg-background border border-border text-foreground-muted italic">
            Passage snippet indexed in evidence store.
          </div>
        )}

        {/* Retrieval Provenance */}
        <div>
          <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5">
            Retrieval Provenance
          </div>
          <div className="divide-y divide-border border border-border rounded bg-background">
            <div className="flex justify-between py-2 px-3">
              <span className="text-foreground-secondary">Source Type</span>
              <span className="font-mono text-foreground">
                {isWebSource ? "External Web Research" : "Local Vector Store"}
              </span>
            </div>
            {isWebSource ? (
              <>
                {source.domain && (
                  <div className="flex justify-between py-2 px-3">
                    <span className="text-foreground-secondary">Domain</span>
                    <span className="font-mono text-foreground">{source.domain}</span>
                  </div>
                )}
                <div className="flex justify-between py-2 px-3">
                  <span className="text-foreground-secondary">Evidence Extraction</span>
                  <span className="font-mono text-foreground">Groq Research Agent</span>
                </div>
              </>
            ) : (
              <>
                <div className="flex justify-between py-2 px-3">
                  <span className="text-foreground-secondary">Method</span>
                  <span className="font-mono text-foreground">
                    {source.strategy || "Hybrid (Dense + BM25)"}
                  </span>
                </div>
                <div className="flex justify-between py-2 px-3">
                  <span className="text-foreground-secondary">Reranking</span>
                  <span className="font-mono text-foreground">FlashRank Bi-Encoder</span>
                </div>
                <div className="flex justify-between py-2 px-3">
                  <span className="text-foreground-secondary">Index Location</span>
                  <span className="font-mono text-foreground">ChromaDB / chunks</span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Primary Action Button */}
        <div className="pt-2">
          {isWebSource && source.url ? (
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block"
            >
              <Button variant="secondary" size="sm" className="w-full justify-center">
                <ExternalLink className="w-3.5 h-3.5" />
                <span>Visit source website →</span>
              </Button>
            </a>
          ) : (
            <Link href="/documents">
              <Button variant="secondary" size="sm" className="w-full justify-center">
                <ExternalLink className="w-3.5 h-3.5" />
                <span>View in Document Library</span>
              </Button>
            </Link>
          )}
        </div>
      </div>
    </Sheet>
  );
};
