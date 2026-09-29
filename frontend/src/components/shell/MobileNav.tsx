"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Menu, X } from "lucide-react";
import { Sidebar } from "./Sidebar";

export const MobileNav: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="md:hidden flex items-center justify-between px-4 py-3 bg-surface border-b border-border sticky top-0 z-30">
      <Link
        href="/chat"
        className="flex items-center gap-2 text-foreground font-semibold text-sm tracking-tight"
      >
        <span className="w-2.5 h-2.5 rounded-full bg-accent inline-block" />
        <span>LOCALMIND</span>
      </Link>

      <button
        onClick={() => setIsOpen(true)}
        aria-label="Open navigation menu"
        className="p-1.5 rounded text-foreground-secondary hover:text-foreground hover:bg-surface-hover"
      >
        <Menu className="w-5 h-5" />
      </button>

      {/* Slide-out mobile drawer */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex">
          {/* Backdrop */}
          <div
            onClick={() => setIsOpen(false)}
            className="fixed inset-0 bg-black/70 backdrop-blur-xs transition-opacity"
          />

          {/* Drawer Panel */}
          <div className="relative w-64 max-w-[80vw] h-full bg-surface shadow-2xl flex flex-col z-10 animate-in slide-in-from-left duration-200">
            <div className="absolute right-2 top-3 z-20">
              <button
                onClick={() => setIsOpen(false)}
                aria-label="Close menu"
                className="p-1.5 rounded text-foreground-muted hover:text-foreground hover:bg-surface-hover"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <Sidebar onCloseMobile={() => setIsOpen(false)} />
          </div>
        </div>
      )}
    </header>
  );
};
