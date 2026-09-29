"use client";

import React, { useState, useRef, useEffect } from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface ChatInputProps {
  onSend: (query: string, useWebResearch: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  disabled,
  placeholder = "Ask your knowledge base...",
}) => {
  const [value, setValue] = useState("");
  const [useWebResearch, setUseWebResearch] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        160
      )}px`;
    }
  }, [value]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, useWebResearch);
    setValue("");
    setUseWebResearch(false); // Reset per-turn state
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-4">
      <form
        onSubmit={handleSubmit}
        className="relative flex items-end gap-2 p-2 rounded-xl bg-surface border border-border focus-within:border-accent transition-colors duration-150 shadow-lg"
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="flex-1 bg-transparent border-0 text-sm text-foreground placeholder:text-foreground-muted resize-none focus:outline-hidden py-1 px-2 leading-relaxed max-h-40 disabled:opacity-50"
        />

        <div className="flex items-center gap-1.5 shrink-0 pb-0.5">
          <button
            type="button"
            onClick={() => setUseWebResearch((prev) => !prev)}
            aria-pressed={useWebResearch}
            disabled={disabled}
            title={useWebResearch ? "Web Research enabled" : "Web Research disabled"}
            className={`inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-colors duration-150 border select-none cursor-pointer ${
              useWebResearch
                ? "bg-accent/15 text-accent-text border-accent/40 font-medium"
                : "bg-surface hover:bg-surface-hover text-foreground-muted hover:text-foreground-secondary border-border"
            }`}
          >
            <span>Web Research</span>
            <span
              className={`w-1.5 h-1.5 rounded-full transition-colors ${
                useWebResearch ? "bg-accent" : "border border-foreground-muted bg-transparent"
              }`}
            />
          </button>

          <Button
            type="submit"
            disabled={!value.trim() || disabled}
            variant="primary"
            size="sm"
            className="rounded-lg p-2 shrink-0 h-8 w-8"
            title="Send query (Enter)"
          >
            <ArrowUp className="w-4 h-4" />
          </Button>
        </div>
      </form>
      <div className="text-[11px] text-foreground-muted text-center mt-2">
        Enter to send · Shift + Enter for new line
      </div>
    </div>
  );
};
