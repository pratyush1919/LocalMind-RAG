"use client";

import React, { useEffect, useRef, useState, useMemo, useCallback } from "react";
import * as d3 from "d3-force";
import { GraphEdge, GraphNode } from "@/lib/types";

interface SimulationNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  frequency?: number;
  x?: number;
  y?: number;
}

interface SimulationLink extends d3.SimulationLinkDatum<SimulationNode> {
  source: string | SimulationNode;
  target: string | SimulationNode;
  predicate: string;
}

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNodeId: string | null;
  onSelectNode: (node: GraphNode | null) => void;
  focusHops?: number;
  zoomLevel: number;
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({
  nodes,
  edges,
  selectedNodeId,
  onSelectNode,
  focusHops = 1,
  zoomLevel,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });

  // Update container dimensions on resize
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth || 800,
          height: containerRef.current.clientHeight || 600,
        });
      }
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  // Compute connected neighbors for selected node based on focusHops
  const neighborIds = useMemo(() => {
    if (!selectedNodeId) return new Set<string>();
    const neighbors = new Set<string>([selectedNodeId]);

    let currentLayer = new Set<string>([selectedNodeId]);
    for (let hop = 0; hop < focusHops; hop++) {
      const nextLayer = new Set<string>();
      for (const edge of edges) {
        const u = typeof edge.source === "object" ? (edge.source as SimulationNode).id : edge.source;
        const v = typeof edge.target === "object" ? (edge.target as SimulationNode).id : edge.target;

        if (currentLayer.has(u)) {
          neighbors.add(v);
          nextLayer.add(v);
        }
        if (currentLayer.has(v)) {
          neighbors.add(u);
          nextLayer.add(u);
        }
      }
      currentLayer = nextLayer;
    }
    return neighbors;
  }, [selectedNodeId, edges, focusHops]);

  // Clone nodes and edges for simulation
  const [simNodes, setSimNodes] = useState<SimulationNode[]>([]);
  const [simEdges, setSimEdges] = useState<SimulationLink[]>([]);

  useEffect(() => {
    if (nodes.length === 0) {
      setSimNodes([]);
      setSimEdges([]);
      return;
    }

    const nList: SimulationNode[] = nodes.map((n) => ({
      ...n,
      x: dimensions.width / 2 + (Math.random() - 0.5) * 200,
      y: dimensions.height / 2 + (Math.random() - 0.5) * 200,
    }));

    const eList: SimulationLink[] = edges.map((e) => ({
      source: e.source,
      target: e.target,
      predicate: e.predicate,
    }));

    const simulation = d3
      .forceSimulation(nList)
      .force(
        "link",
        d3
          .forceLink<SimulationNode, SimulationLink>(eList)
          .id((d) => d.id)
          .distance(90)
      )
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(dimensions.width / 2, dimensions.height / 2))
      .force("collision", d3.forceCollide().radius(25))
      .stop();

    // Run simulation synchronously for ~120 ticks for immediate stable layout
    for (let i = 0; i < 120; ++i) simulation.tick();

    setSimNodes([...nList]);
    setSimEdges([...eList]);
  }, [nodes, edges, dimensions.width, dimensions.height]);

  // Handle pan dragging
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Left click only
    setIsDragging(true);
    dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.current.x,
      y: e.clientY - dragStart.current.y,
    });
  }, [isDragging]);

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      className="w-full h-full relative overflow-hidden select-none bg-background cursor-grab active:cursor-grabbing"
    >
      <svg
        className="w-full h-full"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoomLevel})`,
          transformOrigin: "center center",
          transition: isDragging ? "none" : "transform 150ms ease-out",
        }}
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="20"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="var(--color-border-active)" />
          </marker>
        </defs>

        {/* Quiet Relationship Edges */}
        <g className="edges">
          {simEdges.map((edge, idx) => {
            const src = edge.source as SimulationNode;
            const tgt = edge.target as SimulationNode;
            if (src.x === undefined || src.y === undefined || tgt.x === undefined || tgt.y === undefined) {
              return null;
            }

            const isRelated =
              !selectedNodeId ||
              (neighborIds.has(src.id) && neighborIds.has(tgt.id));

            return (
              <line
                key={`edge-${idx}`}
                x1={src.x}
                y1={src.y}
                x2={tgt.x}
                y2={tgt.y}
                stroke={isRelated ? "var(--color-border-active)" : "var(--color-border-subtle)"}
                strokeWidth={isRelated && selectedNodeId ? 1.5 : 1}
                strokeOpacity={isRelated ? 0.7 : 0.15}
                markerEnd="url(#arrow)"
              />
            );
          })}
        </g>

        {/* Understated Entity Nodes */}
        <g className="nodes">
          {simNodes.map((node) => {
            if (node.x === undefined || node.y === undefined) return null;

            const isSelected = selectedNodeId === node.id;
            const isNeighbor = neighborIds.has(node.id);
            const isFaded = selectedNodeId && !isNeighbor;

            const radius = Math.min(18, Math.max(9, 9 + (node.frequency || 0) * 1.5));

            return (
              <g
                key={`node-${node.id}`}
                transform={`translate(${node.x}, ${node.y})`}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectNode(isSelected ? null : { id: node.id, label: node.label, frequency: node.frequency });
                }}
                className="cursor-pointer"
                opacity={isFaded ? 0.2 : 1}
              >
                {/* Selection ring */}
                {isSelected && (
                  <circle
                    r={radius + 4}
                    fill="none"
                    stroke="var(--color-accent)"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                    className="animate-pulse"
                  />
                )}

                {/* Node Body */}
                <circle
                  r={radius}
                  fill={isSelected ? "var(--color-accent-hover)" : "var(--color-bg-surface)"}
                  stroke={isSelected ? "var(--color-accent)" : "var(--color-border-active)"}
                  strokeWidth={1.5}
                  className="transition-colors duration-150 hover:stroke-accent"
                />

                {/* Node Label (understated, crisp) */}
                <text
                  dy={radius + 13}
                  textAnchor="middle"
                  className="text-[11px] font-mono fill-foreground-secondary pointer-events-none tracking-tight select-none"
                  fontWeight={isSelected ? "600" : "400"}
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
};
