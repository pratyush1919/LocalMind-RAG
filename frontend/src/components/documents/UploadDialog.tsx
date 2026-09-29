"use client";

import React, { useState, useRef } from "react";
import { Upload, ChevronDown, ChevronRight, FileCheck, AlertCircle, Loader2 } from "lucide-react";
import { Modal } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { api } from "@/lib/api";

interface UploadDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".md"];
const MAX_SIZE_MB = 25;

export const UploadDialog: React.FC<UploadDialogProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [strategy, setStrategy] = useState<"sentence" | "parent_child" | "semantic">("sentence");
  const [docCategory, setDocCategory] = useState<string>("");
  const [buildGraph, setBuildGraph] = useState<boolean>(false);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  // States: idle | uploading | processing | success | error
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "processing" | "success" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetForm = () => {
    setSelectedFile(null);
    setUploadStatus("idle");
    setErrorMessage(null);
  };

  const handleFileChange = (file: File) => {
    setErrorMessage(null);
    const ext = `.${file.name.split(".").pop()?.toLowerCase()}`;
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMessage(`Unsupported format. Supported: ${ALLOWED_EXTENSIONS.join(", ")}`);
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMessage(`File exceeds the ${MAX_SIZE_MB} MB limit.`);
      return;
    }
    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      setUploadStatus("uploading");
      setErrorMessage(null);

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("strategy", strategy);
      if (docCategory.trim()) {
        formData.append("doc_category", docCategory.trim());
      }
      formData.append("build_graph", String(buildGraph));

      setUploadStatus("processing");
      const res = await api.uploadDocument(formData);

      if (res.status === "success") {
        setUploadStatus("success");
        setTimeout(() => {
          resetForm();
          onSuccess();
          onClose();
        }, 1000);
      } else {
        setUploadStatus("error");
        setErrorMessage(res.message || "Failed to ingest document.");
      }
    } catch (err: unknown) {
      setUploadStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Upload and ingestion failed.");
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => {
        if (uploadStatus !== "processing" && uploadStatus !== "uploading") {
          resetForm();
          onClose();
        }
      }}
      title="Add documents"
      description="Add knowledge to your local workspace. Indexed chunks are immediately searchable."
    >
      <div className="space-y-4">
        {/* Dropzone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-border hover:border-border-active rounded-lg p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors bg-background"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.doc,.txt,.md"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.[0]) handleFileChange(e.target.files[0]);
            }}
          />

          <Upload className="w-6 h-6 text-foreground-muted mb-2" />
          <div className="text-xs font-medium text-foreground">
            {selectedFile ? selectedFile.name : "Drop file here or browse"}
          </div>
          <div className="text-[11px] text-foreground-muted mt-1">
            PDF · DOCX · DOC · TXT · MD (Max 25 MB)
          </div>
        </div>

        {/* Selected File Notice */}
        {selectedFile && (
          <div className="flex items-center justify-between p-2.5 rounded bg-surface border border-border text-xs">
            <span className="font-mono text-foreground truncate">{selectedFile.name}</span>
            <span className="text-foreground-muted text-[11px] shrink-0">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </span>
          </div>
        )}

        {/* Advanced Options Accordion */}
        <div className="border-t border-border pt-2">
          <button
            type="button"
            onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
            className="flex items-center gap-1.5 text-xs text-foreground-secondary hover:text-foreground py-1"
          >
            {isAdvancedOpen ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            <span>Advanced options</span>
          </button>

          {isAdvancedOpen && (
            <div className="mt-2.5 space-y-3 pl-2 text-xs">
              <div>
                <label className="block text-[11px] font-medium text-foreground-secondary mb-1">
                  Chunking Strategy
                </label>
                <div className="flex gap-4">
                  {(["sentence", "parent_child", "semantic"] as const).map((strat) => (
                    <label key={strat} className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="radio"
                        name="strategy"
                        value={strat}
                        checked={strategy === strat}
                        onChange={() => setStrategy(strat)}
                        className="accent-accent"
                      />
                      <span className="capitalize">{strat.replace("_", " ")}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-[11px] font-medium text-foreground-secondary mb-1">
                  Document Category (optional)
                </label>
                <Input
                  value={docCategory}
                  onChange={(e) => setDocCategory(e.target.value)}
                  placeholder="e.g. technical, research_paper, report"
                />
              </div>

              <div className="pt-1">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="build_graph"
                    checked={buildGraph}
                    onChange={(e) => setBuildGraph(e.target.checked)}
                    className="accent-accent rounded"
                  />
                  <label htmlFor="build_graph" className="cursor-pointer text-foreground-secondary">
                    Extract knowledge graph triples using local LLM
                  </label>
                </div>
                <p className="text-[11px] text-foreground-muted pl-5 pt-0.5">
                  Analyzes representative document sections to extract entity relationships. Processing time increases with local LLM inference.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Real Status Feedback */}
        {uploadStatus === "uploading" && (
          <div className="flex items-center gap-2 text-xs text-foreground-secondary py-1">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
            <span>Uploading file to local staging...</span>
          </div>
        )}
        {uploadStatus === "processing" && (
          <div className="flex items-center gap-2 text-xs text-foreground-secondary py-1">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
            <span>
              {buildGraph
                ? "Indexing document vectors and extracting relational graph triples (local LLM)..."
                : "Parsing structure, generating chunk embeddings & updating BM25..."}
            </span>
          </div>
        )}
        {uploadStatus === "success" && (
          <div className="flex items-center gap-2 text-xs text-success py-1">
            <FileCheck className="w-4 h-4" />
            <span>Document successfully indexed!</span>
          </div>
        )}
        {uploadStatus === "error" && errorMessage && (
          <div className="flex items-center gap-2 text-xs text-danger py-1">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-2 pt-2 border-t border-border">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              resetForm();
              onClose();
            }}
            disabled={uploadStatus === "uploading" || uploadStatus === "processing"}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={handleUpload}
            disabled={!selectedFile || uploadStatus === "uploading" || uploadStatus === "processing"}
          >
            {uploadStatus === "uploading" || uploadStatus === "processing" ? (
              <span className="flex items-center gap-1.5">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Ingesting...</span>
              </span>
            ) : (
              "Add documents"
            )}
          </Button>
        </div>
      </div>
    </Modal>
  );
};
