"use client";

import React, { useState, useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { MessageItem, ChatMessageUI } from "./MessageItem";
import { ChatInput } from "./ChatInput";
import { SourceInspector, SourceEvidence } from "./SourceInspector";
import { streamChatQuery } from "@/lib/sse";
import { api } from "@/lib/api";
import { ChatResponse, ExecutionEvent, ExecutionStage } from "@/lib/types";

export const ChatWorkspace: React.FC = () => {
  const searchParams = useSearchParams();
  const sessionParam = searchParams.get("session");

  const [messages, setMessages] = useState<ChatMessageUI[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(sessionParam);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedSource, setSelectedSource] = useState<SourceEvidence | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll when messages update
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  // Load session history when URL param changes
  useEffect(() => {
    if (sessionParam) {
      setActiveSessionId(sessionParam);
      loadSessionHistory(sessionParam);
    } else {
      setActiveSessionId(null);
      setMessages([]);
    }
  }, [sessionParam]);

  const loadSessionHistory = async (sessionId: string) => {
    try {
      setError(null);
      const detail = await api.getSessionHistory(sessionId);
      if (detail && detail.messages) {
        const mapped: ChatMessageUI[] = detail.messages.map((m, idx) => ({
          id: `${sessionId}-${idx}`,
          role: m.role,
          content: m.content,
          citations: m.citations || [],
          consensus_score: m.consensus_score,
        }));
        setMessages(mapped);
      }
    } catch (err) {
      console.error("Failed to load session history:", err);
      setError("Could not load previous conversation history.");
    }
  };

  const handleSend = (query: string, useWebResearch: boolean = false) => {
    if (isStreaming) return;
    setError(null);

    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now()}`;

    const userMessage: ChatMessageUI = {
      id: userMsgId,
      role: "user",
      content: query,
    };

    const assistantMessage: ChatMessageUI = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
      executionEvents: [],
      sources: [],
      citations: [],
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    const eventsList: ExecutionEvent[] = [];

    streamChatQuery(
      {
        query,
        session_id: activeSessionId,
        user_role: "analyst",
        use_memory: true,
        use_web_research: useWebResearch,
      },
      {
        onEvent: (event: ExecutionStage, data: Record<string, unknown>) => {
          eventsList.push({ event, data, timestamp: Date.now() });

          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    executionEvents: [...eventsList],
                  }
                : m
            )
          );
        },
        onFinal: (response: ChatResponse) => {
          setIsStreaming(false);

          if (!activeSessionId && response.session_id) {
            setActiveSessionId(response.session_id);
            // Update URL without full refresh
            window.history.replaceState(
              {},
              "",
              `/chat?session=${encodeURIComponent(response.session_id)}`
            );
          }

          // Trigger sidebar refresh for the new conversation
          window.dispatchEvent(new Event("localmind:refresh-sessions"));

          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    content: response.answer,
                    isStreaming: false,
                    citations: response.citations || [],
                    sources: response.sources || [],
                    consensus_score: response.consensus_score,
                    metadata: response,
                    guardrail_blocked: response.guardrail_blocked,
                    guardrail_reason: response.guardrail_reason,
                    usedMemory: true,
                  }
                : m
            )
          );
        },
        onError: (err: Error) => {
          setIsStreaming(false);
          setError(err.message || "An error occurred while researching your question.");
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    isStreaming: false,
                    content:
                      "Something went wrong while processing your query. Please check local system status and try again.",
                  }
                : m
            )
          );
        },
      }
    );
  };

  const handleSelectCitation = (evidence: SourceEvidence) => {
    setSelectedSource(evidence);
    setIsInspectorOpen(true);
  };

  return (
    <div className="flex-1 h-full flex flex-col relative overflow-hidden bg-background">
      {/* Scrollable Conversation Container */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto px-4 py-6 flex flex-col items-center"
      >
        <div className="w-full max-w-3xl space-y-4">
          {/* EMPTY STATE */}
          {messages.length === 0 && (
            <div className="min-h-[50vh] flex flex-col justify-center items-center text-center px-4 max-w-lg mx-auto py-12">
              <div className="space-y-2 mb-8">
                <h1 className="text-xl sm:text-2xl font-semibold text-foreground tracking-tight">
                  LocalMind
                </h1>
                <p className="text-sm font-medium text-foreground-secondary">
                  Search your local knowledge.
                </p>
                <p className="text-xs text-foreground-muted leading-relaxed max-w-md">
                  Ask questions about your indexed documents and explore what your knowledge base contains.
                </p>
              </div>

              {/* Restrained Prompt Starters */}
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "What is hybrid retrieval and how does it work?",
                  "Compare Java platform overview with Python notes",
                  "What are the main principles of quantum computing?",
                ].map((promptText, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(promptText)}
                    className="text-xs px-3 py-1.5 rounded bg-surface hover:bg-surface-hover text-foreground-secondary hover:text-foreground border border-border transition-colors text-left"
                  >
                    {promptText}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* MESSAGES LIST */}
          {messages.map((msg) => (
            <MessageItem
              key={msg.id}
              message={msg}
              onSelectCitation={handleSelectCitation}
            />
          ))}

          {/* ERROR BANNER */}
          {error && (
            <div className="p-3 rounded bg-surface border border-danger/40 text-xs text-danger flex items-center justify-between">
              <span>{error}</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* FIXED BOTTOM CHAT INPUT */}
      <div className="shrink-0 border-t border-border/40 pt-3 bg-background/80 backdrop-blur-xs">
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>

      {/* SOURCE EVIDENCE INSPECTOR (DRAWER/SHEET) */}
      <SourceInspector
        source={selectedSource}
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
      />
    </div>
  );
};
