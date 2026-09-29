import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "text" | "rectangular" | "circular";
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className,
  variant = "rectangular",
  ...props
}) => {
  const base = "animate-pulse bg-surface-hover/80";

  const variants = {
    text: "h-3.5 w-full rounded-sm",
    rectangular: "rounded",
    circular: "rounded-full",
  };

  return (
    <div
      className={twMerge(clsx(base, variants[variant], className))}
      {...props}
    />
  );
};
