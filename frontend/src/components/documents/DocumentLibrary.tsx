"use client";

import React, { useState, useEffect } from "react";
import { Plus, Search, FileText } from "lucide-react";
import { DocumentItem, DocumentListResponse } from "@/lib/types";
import { api } from "@/lib/api";
import { DocumentRow } from "./DocumentRow";
import { DocumentInspector } from "./DocumentInspector";
import { UploadDialog } from "./UploadDialog";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Skeleton } from "@/components/ui/Skeleton";

export const DocumentLibrary: React.FC = () => {
  const [data, setData] = useState<DocumentListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFormat, setSelectedFormat] = useState<string>("all");
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const res = await api.listDocuments();
      setData(res);
    } catch (err) {
      console.error("Failed to load documents:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const documents = data?.documents || [];

  const filteredDocs = documents.filter((doc) => {
    const matchesSearch =
      doc.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (doc.doc_category && doc.doc_category.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesFormat =
      selectedFormat === "all" || doc.file_type.toLowerCase() === selectedFormat.toLowerCase();

    return matchesSearch && matchesFormat;
  });

  const formats = ["all", "pdf", "docx", "txt", "md"];

  return (
    <div className="flex-1 h-full overflow-y-auto px-6 py-6 max-w-5xl mx-auto w-full flex flex-col space-y-6">
      {/* Workspace Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-5">
        <div>
          <h1 className="text-xl sm:text-2xl font-semibold text-foreground tracking-tight">
            Documents
          </h1>
          <p className="text-xs text-foreground-secondary mt-1">
            Your local knowledge base
          </p>
          <div className="text-xs text-foreground-muted font-mono mt-0.5">
            {data ? (
              <span>
                {data.total_documents} document{data.total_documents !== 1 ? "s" : ""} ·{" "}
                {data.total_chunks} indexed chunks
              </span>
            ) : (
              <span>Scanning local storage...</span>
            )}
          </div>
        </div>

        <Button
          onClick={() => setIsUploadOpen(true)}
          variant="primary"
          size="sm"
          className="gap-1.5 self-start sm:self-auto"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Add documents</span>
        </Button>
      </div>

      {/* Search & Filter Controls */}
      <div className="flex flex-col sm:flex-row items-center gap-3">
        <div className="relative flex-1 w-full">
          <Search className="w-4 h-4 text-foreground-muted absolute left-3 top-2.5" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search documents or categories..."
            className="pl-9 text-xs"
          />
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto w-full sm:w-auto self-start">
          {formats.map((fmt) => (
            <button
              key={fmt}
              onClick={() => setSelectedFormat(fmt)}
              className={`px-2.5 py-1 rounded text-xs font-mono uppercase transition-colors ${
                selectedFormat === fmt
                  ? "bg-surface-active text-foreground font-medium border border-border-active"
                  : "bg-surface hover:bg-surface-hover text-foreground-secondary border border-border"
              }`}
            >
              {fmt}
            </button>
          ))}
        </div>
      </div>

      {/* Documents List */}
      <div className="space-y-2 flex-1">
        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-16 w-full rounded-lg" />
            ))}
          </div>
        ) : filteredDocs.length > 0 ? (
          filteredDocs.map((doc) => (
            <DocumentRow
              key={doc.filename}
              document={doc}
              isSelected={selectedDoc?.filename === doc.filename}
              onSelect={() => setSelectedDoc(doc)}
            />
          ))
        ) : (
          /* EMPTY STATE */
          <div className="py-16 text-center border border-dashed border-border rounded-xl p-8 flex flex-col items-center justify-center space-y-3">
            <div className="p-3 rounded-full bg-surface border border-border">
              <FileText className="w-6 h-6 text-foreground-muted" />
            </div>
            <div>
              <div className="text-sm font-medium text-foreground">Nothing here yet.</div>
              <p className="text-xs text-foreground-muted mt-1 max-w-sm">
                Add documents to give LocalMind something to search.
              </p>
            </div>
            <Button
              onClick={() => setIsUploadOpen(true)}
              variant="secondary"
              size="sm"
              className="gap-1.5 mt-2"
            >
              <Plus className="w-3.5 h-3.5 text-accent" />
              <span>Add documents</span>
            </Button>
          </div>
        )}
      </div>

      {/* Document Inspector Sheet */}
      <DocumentInspector
        document={selectedDoc}
        isOpen={!!selectedDoc}
        onClose={() => setSelectedDoc(null)}
        onDeleted={() => {
          fetchDocuments();
          setSelectedDoc(null);
        }}
      />

      {/* Upload Dialog */}
      <UploadDialog
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSuccess={fetchDocuments}
      />
    </div>
  );
};
