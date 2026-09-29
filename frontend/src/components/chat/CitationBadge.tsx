"use client";

import React from "react";

interface CitationBadgeProps {
  label: string;
  onClick: () => void;
}

export const CitationBadge: React.FC<CitationBadgeProps> = ({ label, onClick }) => {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className="inline-flex items-center justify-center px-1.5 py-0.5 rounded text-[11px] font-mono text-accent-text bg-accent-subtle border border-accent/20 hover:bg-accent/20 transition-colors cursor-pointer align-baseline mx-0.5"
      title={`Inspect evidence for ${label}`}
    >
      {label}
    </button>
  );
};
