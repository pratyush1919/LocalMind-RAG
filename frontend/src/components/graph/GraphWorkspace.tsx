"use client";

import React, { useState, useEffect } from "react";
import { Search, Network, ArrowRight } from "lucide-react";
import { GraphData, GraphNode } from "@/lib/types";
import { api } from "@/lib/api";
import { GraphCanvas } from "./GraphCanvas";
import { GraphControls } from "./GraphControls";
import { NodeInspector } from "./NodeInspector";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import Link from "next/link";

export const GraphWorkspace: React.FC = () => {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [zoom, setZoom] = useState(1.0);
  const [focusHops, setFocusHops] = useState(1);
  const [error, setError] = useState<string | null>(null);

  const fetchGraph = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getGraphData();
      setGraphData(data);
    } catch (err) {
      console.error("Failed to load knowledge graph:", err);
      setError("Could not load the knowledge graph. Documents are still available.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraph();
  }, []);

  // Filter search matches
  const searchMatches = (graphData?.nodes || []).filter(
    (n) =>
      searchQuery.trim().length > 0 &&
      (n.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
        n.id.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const handleSelectNode = (node: GraphNode | null) => {
    setSelectedNode(node);
    if (node) {
      setSearchQuery("");
    }
  };

  const handleSelectNodeById = (nodeId: string) => {
    const found = (graphData?.nodes || []).find((n) => n.id === nodeId);
    if (found) {
      setSelectedNode(found);
    }
  };

  const isEmpty = !loading && (!graphData || graphData.total_nodes === 0);

  return (
    <div className="flex-1 h-full flex flex-col relative overflow-hidden bg-background">
      {/* Workspace Header */}
      <div className="p-4 sm:p-5 border-b border-border bg-surface/50 backdrop-blur-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3 z-10">
        <div>
          <h1 className="text-xl font-semibold text-foreground tracking-tight">
            Knowledge Graph
          </h1>
          <p className="text-xs text-foreground-secondary mt-0.5">
            Explore relationships across your knowledge base
          </p>
          <div className="text-xs text-foreground-muted font-mono mt-0.5">
            {graphData ? (
              <span>
                {graphData.total_nodes} nodes · {graphData.total_edges} relationships
              </span>
            ) : (
              <span>Loading graph structure...</span>
            )}
          </div>
        </div>

        {/* Search concepts/entities */}
        <div className="relative w-full sm:w-72">
          <Search className="w-3.5 h-3.5 text-foreground-muted absolute left-3 top-2.5" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search concepts, entities..."
            className="pl-8 text-xs h-8"
          />

          {/* Search Dropdown Matches */}
          {searchMatches.length > 0 && (
            <div className="absolute top-10 left-0 right-0 max-h-48 overflow-y-auto bg-surface border border-border rounded-lg shadow-xl z-20 divide-y divide-border">
              {searchMatches.map((n) => (
                <button
                  key={n.id}
                  onClick={() => handleSelectNode(n)}
                  className="w-full text-left px-3 py-2 text-xs text-foreground hover:bg-surface-hover flex items-center justify-between"
                >
                  <span className="font-medium">{n.label}</span>
                  <span className="text-[10px] text-foreground-muted font-mono">
                    {n.frequency || 1} mentions
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 relative overflow-hidden">
        {loading ? (
          <div className="h-full flex flex-col items-center justify-center text-xs text-foreground-muted space-y-2">
            <Network className="w-8 h-8 text-foreground-muted animate-pulse" />
            <span>Loading knowledge graph...</span>
          </div>
        ) : error ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
            <div className="text-sm font-medium text-foreground">{error}</div>
            <Button variant="secondary" size="sm" onClick={fetchGraph}>
              Retry
            </Button>
          </div>
        ) : isEmpty ? (
          /* EMPTY STATE */
          <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-3">
            <div className="p-3 rounded-full bg-surface border border-border">
              <Network className="w-6 h-6 text-foreground-muted" />
            </div>
            <div>
              <div className="text-sm font-medium text-foreground">
                Your knowledge graph is empty.
              </div>
              <p className="text-xs text-foreground-muted mt-1 max-w-sm">
                Build a graph from your indexed documents to explore relationships between concepts.
              </p>
            </div>
            <Link href="/documents">
              <Button variant="secondary" size="sm" className="gap-1.5 mt-2">
                <span>Go to Documents</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>
          </div>
        ) : (
          <>
            {/* Force Simulation Canvas */}
            <GraphCanvas
              nodes={graphData?.nodes || []}
              edges={graphData?.edges || []}
              selectedNodeId={selectedNode?.id || null}
              onSelectNode={handleSelectNode}
              focusHops={focusHops}
              zoomLevel={zoom}
            />

            {/* Floating Zoom & Focus Controls */}
            <div className="absolute bottom-4 right-4 z-10">
              <GraphControls
                zoom={zoom}
                onZoomIn={() => setZoom((z) => Math.min(2.5, z + 0.15))}
                onZoomOut={() => setZoom((z) => Math.max(0.4, z - 0.15))}
                onResetZoom={() => setZoom(1.0)}
                focusHops={focusHops}
                onFocusHopsChange={setFocusHops}
              />
            </div>
          </>
        )}
      </div>

      {/* Node Inspector Sheet */}
      <NodeInspector
        node={selectedNode}
        edges={graphData?.edges || []}
        isOpen={!!selectedNode}
        onClose={() => setSelectedNode(null)}
        onSelectNode={handleSelectNodeById}
      />
    </div>
  );
};
