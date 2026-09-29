"use client";

import React, { useState } from "react";
import { FileText, CheckCircle2, Trash2 } from "lucide-react";
import { DocumentItem } from "@/lib/types";
import { Sheet } from "@/components/ui/Sheet";
import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";

import { api } from "@/lib/api";

interface DocumentInspectorProps {
  document: DocumentItem | null;
  isOpen: boolean;
  onClose: () => void;
  onDeleted?: (filename: string) => void;
}

export const DocumentInspector: React.FC<DocumentInspectorProps> = ({
  document,
  isOpen,
  onClose,
  onDeleted,
}) => {
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  if (!document) return null;

  const handleDelete = async () => {
    try {
      setIsDeleting(true);
      setDeleteError(null);
      await api.deleteDocument(document.filename);
      setIsDeleteModalOpen(false);
      onClose();
      if (onDeleted) onDeleted(document.filename);
    } catch (err: unknown) {
      console.error("Failed to delete document:", err);
      setDeleteError(err instanceof Error ? err.message : "Failed to delete document from knowledge base.");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <>
      <Sheet
        isOpen={isOpen}
        onClose={onClose}
        title={document.filename}
        subtitle="Document metadata and status"
      >
        <div className="space-y-4 text-xs">
          {/* Status Overview */}
          <div className="p-3 rounded bg-background border border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-foreground-secondary" />
              <div>
                <div className="font-medium text-foreground">{document.filename}</div>
                <div className="text-[11px] text-foreground-muted uppercase font-mono">
                  {document.file_type} Document
                </div>
              </div>
            </div>
            <span className="flex items-center gap-1 text-xs text-success font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Indexed</span>
            </span>
          </div>

          {/* Properties Table */}
          <div>
            <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5">
              Properties
            </div>
            <div className="divide-y divide-border border border-border rounded bg-background">
              <div className="flex justify-between py-2 px-3">
                <span className="text-foreground-secondary">Chunks Generated</span>
                <span className="font-mono text-foreground">{document.chunk_count}</span>
              </div>
              <div className="flex justify-between py-2 px-3">
                <span className="text-foreground-secondary">Category</span>
                <span className="font-mono text-foreground capitalize">
                  {document.doc_category || "General"}
                </span>
              </div>
              <div className="flex justify-between py-2 px-3">
                <span className="text-foreground-secondary">Format</span>
                <span className="font-mono text-foreground uppercase">{document.file_type}</span>
              </div>
              {document.created_at && (
                <div className="flex justify-between py-2 px-3">
                  <span className="text-foreground-secondary">Indexed At</span>
                  <span className="font-mono text-foreground">
                    {new Date(document.created_at).toLocaleDateString()}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Indexing Details */}
          <div>
            <div className="text-[11px] font-medium text-foreground-muted uppercase tracking-wider mb-1.5">
              Storage Locations
            </div>
            <div className="p-3 rounded bg-background border border-border space-y-1 text-[11px] font-mono text-foreground-secondary">
              <div>Dense Vectors: ChromaDB (chunks)</div>
              <div>Sparse Index: BM25 (vectorstore/bm25)</div>
              <div>Source Nodes: vectorstore/chunks_nodes.pkl</div>
            </div>
          </div>

          {/* Delete Action Button */}
          <div className="pt-4 border-t border-border">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsDeleteModalOpen(true)}
              className="w-full text-danger hover:bg-danger/10 hover:text-danger justify-center gap-2"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Delete document</span>
            </Button>
          </div>
        </div>
      </Sheet>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        title="Delete document?"
        description="This removes the document from your local knowledge base and its indexed content."
      >
        {deleteError && (
          <div className="p-2 rounded bg-danger/10 border border-danger/30 text-danger text-xs">
            {deleteError}
          </div>
        )}
        <div className="flex justify-end gap-2 pt-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setIsDeleteModalOpen(false)}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            variant="danger"
            size="sm"
            onClick={handleDelete}
            disabled={isDeleting}
          >
            {isDeleting ? "Deleting..." : "Delete"}
          </Button>
        </div>
      </Modal>
    </>
  );
};
