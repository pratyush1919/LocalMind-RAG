"use client";

import React from "react";
import { FileText, CheckCircle2 } from "lucide-react";
import { DocumentItem } from "@/lib/types";

interface DocumentRowProps {
  document: DocumentItem;
  isSelected: boolean;
  onSelect: () => void;
}

export const DocumentRow: React.FC<DocumentRowProps> = ({
  document,
  isSelected,
  onSelect,
}) => {
  return (
    <div
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      className={`group flex items-center justify-between p-3 rounded-lg border transition-colors duration-150 cursor-pointer ${
        isSelected
          ? "bg-surface-active border-border-active"
          : "bg-surface hover:bg-surface-hover border-border"
      }`}
    >
      <div className="flex items-center gap-3 min-w-0">
        <div className="p-2 rounded bg-background border border-border shrink-0">
          <FileText className="w-4 h-4 text-foreground-secondary" />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-medium text-foreground truncate">
            {document.filename}
          </div>
          <div className="flex items-center gap-2 text-xs text-foreground-muted mt-0.5">
            <span className="uppercase font-mono text-[10px] px-1.5 py-0.5 rounded bg-background border border-border">
              {document.file_type}
            </span>
            <span>·</span>
            <span>{document.chunk_count} chunks</span>
            {document.doc_category && (
              <>
                <span>·</span>
                <span className="capitalize">{document.doc_category}</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <span className="flex items-center gap-1 text-xs text-success font-medium">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Indexed</span>
        </span>
      </div>
    </div>
  );
};
