import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/shell/AppShell";

export const metadata: Metadata = {
  title: "LocalMind-RAG",
  description: "Private research instrument and local knowledge exploration engine.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-foreground selection:bg-accent/20 selection:text-foreground">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
