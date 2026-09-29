"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Plus,
  MessageSquare,
  FileText,
  Network,
  Settings as SettingsIcon,
  Trash2,
} from "lucide-react";
import { api } from "@/lib/api";
import { SessionSummary } from "@/lib/types";
import { SystemStatus } from "./SystemStatus";
import { Modal } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";

interface SidebarProps {
  onSessionSelect?: (sessionId: string) => void;
  activeSessionId?: string | null;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  onSessionSelect,
  activeSessionId,
  onCloseMobile,
}) => {
  const pathname = usePathname();
  const router = useRouter();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionToDelete, setSessionToDelete] = useState<SessionSummary | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const loadSessions = async () => {
    try {
      const res = await api.listSessions();
      setSessions(res.sessions || []);
    } catch {
      // Quiet fallback
    }
  };

  useEffect(() => {
    loadSessions();
    const handleRefresh = () => loadSessions();
    window.addEventListener("localmind:refresh-sessions", handleRefresh);
    return () => window.removeEventListener("localmind:refresh-sessions", handleRefresh);
  }, []);

  const handleNewChat = () => {
    if (onSessionSelect) {
      onSessionSelect("");
    }
    router.push("/chat");
    if (onCloseMobile) onCloseMobile();
  };

  const handleSelectSession = (sessionId: string) => {
    if (onSessionSelect) {
      onSessionSelect(sessionId);
    }
    router.push(`/chat?session=${encodeURIComponent(sessionId)}`);
    if (onCloseMobile) onCloseMobile();
  };

  const confirmDeleteSession = async () => {
    if (!sessionToDelete) return;
    try {
      setIsDeleting(true);
      await api.deleteSession(sessionToDelete.session_id);
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionToDelete.session_id));
      if (activeSessionId === sessionToDelete.session_id) {
        handleNewChat();
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
    } finally {
      setIsDeleting(false);
      setSessionToDelete(null);
    }
  };

  const navItems = [
    { label: "Chat", href: "/chat", icon: MessageSquare },
    { label: "Documents", href: "/documents", icon: FileText },
    { label: "Knowledge Graph", href: "/graph", icon: Network },
  ];

  return (
    <aside className="w-60 h-full flex flex-col bg-surface border-r border-border select-none">
      {/* Brand Header */}
      <div className="p-4 border-b border-border flex items-center justify-between">
        <Link
          href="/chat"
          onClick={handleNewChat}
          className="flex items-center gap-2 text-foreground font-semibold text-sm tracking-tight hover:opacity-90 transition-opacity"
        >
          <span className="w-2.5 h-2.5 rounded-full bg-accent inline-block" />
          <span>LOCALMIND</span>
        </Link>
      </div>

      {/* Action: + New Chat */}
      <div className="p-3">
        <Button
          onClick={handleNewChat}
          variant="secondary"
          size="sm"
          className="w-full justify-start text-xs font-normal border-border hover:bg-surface-hover"
        >
          <Plus className="w-3.5 h-3.5 text-accent" />
          <span>New chat</span>
        </Button>
      </div>

      {/* Scrollable Navigation Body */}
      <div className="flex-1 overflow-y-auto px-3 py-1 space-y-4">
        {/* RECENT CONVERSATIONS */}
        {sessions.length > 0 && (
          <div>
            <div className="px-2 pb-1.5 text-[11px] font-medium tracking-wider text-foreground-muted uppercase">
              Recent
            </div>
            <div className="space-y-0.5">
              {sessions.slice(0, 15).map((session) => {
                const isActive =
                  pathname === "/chat" &&
                  (activeSessionId === session.session_id ||
                    (typeof window !== "undefined" &&
                      new URLSearchParams(window.location.search).get("session") ===
                        session.session_id));

                return (
                  <div
                    key={session.session_id}
                    className={`group relative flex items-center rounded text-xs transition-colors duration-150 ${
                      isActive
                        ? "bg-surface-active text-foreground font-medium"
                        : "text-foreground-secondary hover:text-foreground hover:bg-surface-hover"
                    }`}
                  >
                    <button
                      onClick={() => handleSelectSession(session.session_id)}
                      className="w-full text-left py-1.5 pl-2.5 pr-7 truncate"
                      title={session.title}
                    >
                      {session.title || "Untitled query"}
                    </button>
                    {/* Unobtrusive delete action on hover */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSessionToDelete(session);
                      }}
                      title="Delete conversation"
                      aria-label="Delete conversation"
                      className="absolute right-1 opacity-0 group-hover:opacity-100 p-1 text-foreground-muted hover:text-danger rounded transition-opacity duration-150"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* WORKSPACE SECTIONS */}
        <div>
          <div className="px-2 pb-1.5 text-[11px] font-medium tracking-wider text-foreground-muted uppercase">
            Workspace
          </div>
          <div className="space-y-0.5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => onCloseMobile && onCloseMobile()}
                  className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded text-xs transition-colors duration-150 ${
                    isActive
                      ? "bg-surface-active text-foreground font-medium"
                      : "text-foreground-secondary hover:text-foreground hover:bg-surface-hover"
                  }`}
                >
                  <Icon
                    className={`w-3.5 h-3.5 ${
                      isActive ? "text-accent" : "text-foreground-muted"
                    }`}
                  />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>

      {/* Bottom Area: Settings + System Status */}
      <div className="p-3 border-t border-border space-y-1">
        <Link
          href="/settings"
          onClick={() => onCloseMobile && onCloseMobile()}
          className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded text-xs transition-colors duration-150 ${
            pathname === "/settings"
              ? "bg-surface-active text-foreground font-medium"
              : "text-foreground-secondary hover:text-foreground hover:bg-surface-hover"
          }`}
        >
          <SettingsIcon className="w-3.5 h-3.5 text-foreground-muted" />
          <span>Settings</span>
        </Link>
        <SystemStatus />
      </div>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={!!sessionToDelete}
        onClose={() => setSessionToDelete(null)}
        title="Delete conversation?"
        description="This removes the conversation history. Your saved knowledge base and memories remain."
      >
        <div className="flex justify-end gap-2 pt-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setSessionToDelete(null)}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            variant="danger"
            size="sm"
            onClick={confirmDeleteSession}
            disabled={isDeleting}
          >
            {isDeleting ? "Deleting..." : "Delete"}
          </Button>
        </div>
      </Modal>
    </aside>
  );
};
