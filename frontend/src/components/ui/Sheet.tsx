import React, { useEffect } from "react";
import { X } from "lucide-react";

export interface SheetProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}

export const Sheet: React.FC<SheetProps> = ({ isOpen, onClose, title, subtitle, children }) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop on mobile */}
      <div
        onClick={onClose}
        className="fixed inset-0 z-40 bg-black/60 md:hidden backdrop-blur-xs transition-opacity"
      />

      {/* Sheet Container: Right side on desktop, bottom drawer on mobile */}
      <aside
        role="dialog"
        aria-modal="true"
        className="fixed z-40 right-0 top-0 bottom-0 w-full sm:w-[420px] bg-surface border-l border-border flex flex-col shadow-2xl transition-transform duration-200 ease-out max-md:top-auto max-md:bottom-0 max-md:max-h-[85vh] max-md:border-t max-md:border-l-0 max-md:rounded-t-xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="min-w-0 flex-1 pr-2">
            <h3 className="text-sm font-semibold text-foreground truncate">{title}</h3>
            {subtitle && (
              <p className="text-xs text-foreground-secondary truncate mt-0.5">{subtitle}</p>
            )}
          </div>
          <button
            onClick={onClose}
            aria-label="Close panel"
            className="p-1 rounded text-foreground-muted hover:text-foreground hover:bg-surface-hover transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">{children}</div>
      </aside>
    </>
  );
};
