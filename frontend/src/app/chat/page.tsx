import React, { Suspense } from "react";
import { ChatWorkspace } from "@/components/chat/ChatWorkspace";

export default function ChatPage() {
  return (
    <Suspense
      fallback={
        <div className="flex-1 h-full flex items-center justify-center text-xs text-foreground-muted">
          Loading conversation...
        </div>
      }
    >
      <ChatWorkspace />
    </Suspense>
  );
}
