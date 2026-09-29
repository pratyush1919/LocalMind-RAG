"use client";

import React from "react";
import { ZoomIn, ZoomOut, RotateCcw, Maximize2 } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface GraphControlsProps {
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
  focusHops: number;
  onFocusHopsChange: (hops: number) => void;
}

export const GraphControls: React.FC<GraphControlsProps> = ({
  zoom,
  onZoomIn,
  onZoomOut,
  onResetZoom,
  focusHops,
  onFocusHopsChange,
}) => {
  return (
    <div className="flex items-center gap-1 p-1 bg-surface border border-border rounded-lg shadow-md text-xs">
      {/* Zoom Out */}
      <button
        onClick={onZoomOut}
        aria-label="Zoom out"
        title="Zoom out"
        className="p-1.5 rounded hover:bg-surface-hover text-foreground-secondary hover:text-foreground transition-colors"
      >
        <ZoomOut className="w-3.5 h-3.5" />
      </button>

      {/* Zoom Level Readout */}
      <span className="font-mono text-[11px] text-foreground-muted px-1.5 select-none">
        {Math.round(zoom * 100)}%
      </span>

      {/* Zoom In */}
      <button
        onClick={onZoomIn}
        aria-label="Zoom in"
        title="Zoom in"
        className="p-1.5 rounded hover:bg-surface-hover text-foreground-secondary hover:text-foreground transition-colors"
      >
        <ZoomIn className="w-3.5 h-3.5" />
      </button>

      {/* Reset Zoom */}
      <button
        onClick={onResetZoom}
        aria-label="Reset zoom"
        title="Reset zoom"
        className="p-1.5 rounded hover:bg-surface-hover text-foreground-secondary hover:text-foreground transition-colors border-l border-border ml-1 pl-2"
      >
        <RotateCcw className="w-3.5 h-3.5" />
      </button>

      {/* Focus Hops Selector */}
      <div className="flex items-center gap-1 border-l border-border pl-2 ml-1 text-xs">
        <span className="text-[11px] text-foreground-muted">Focus:</span>
        <select
          value={focusHops}
          onChange={(e) => onFocusHopsChange(parseInt(e.target.value, 10))}
          className="bg-background border border-border text-foreground text-[11px] rounded px-1.5 py-0.5 focus:outline-hidden font-mono"
        >
          <option value={1}>1 hop</option>
          <option value={2}>2 hops</option>
        </select>
      </div>
    </div>
  );
};
