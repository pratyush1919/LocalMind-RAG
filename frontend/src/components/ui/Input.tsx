import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: boolean;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, error, disabled, ...props }, ref) => {
    return (
      <input
        ref={ref}
        disabled={disabled}
        className={twMerge(
          clsx(
            "w-full bg-background border text-foreground placeholder:text-foreground-muted text-sm rounded px-3 py-2 transition-colors duration-150",
            error
              ? "border-danger focus:border-danger"
              : "border-border hover:border-border-active focus:border-accent",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            className
          )
        )}
        {...props}
      />
    );
  }
);

Input.displayName = "Input";
