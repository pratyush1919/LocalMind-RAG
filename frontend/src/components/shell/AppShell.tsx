"use client";

import React from "react";
import { Sidebar } from "./Sidebar";
import { MobileNav } from "./MobileNav";

interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  return (
    <div className="flex flex-col md:flex-row h-screen w-screen overflow-hidden bg-background">
      {/* Mobile Top Header with Drawer */}
      <MobileNav />

      {/* Persistent Desktop Sidebar */}
      <div className="hidden md:block h-full shrink-0">
        <Sidebar />
      </div>

      {/* Main Workspace Viewport */}
      <main className="flex-1 h-full overflow-hidden flex flex-col relative">
        {children}
      </main>
    </div>
  );
};
