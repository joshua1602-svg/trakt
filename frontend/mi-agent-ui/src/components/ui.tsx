/** Small shared UI primitives (shadcn-flavoured, hand-rolled to avoid deps). */
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Card({
  className,
  children,
  testId,
}: {
  className?: string;
  children: ReactNode;
  testId?: string;
}) {
  return (
    <div
      data-testid={testId}
      className={cn(
        "rounded-xl border border-[var(--color-line)] bg-navy-900/70 backdrop-blur-sm",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: "neutral" | "navy" | "mint" | "amber" | "rose";
  className?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "bg-navy-800 text-ink-300 border-[var(--color-line)]",
    navy: "bg-navy-700/60 text-peri-200 border-navy-600",
    mint: "bg-mint-400/10 text-mint-400 border-mint-400/30",
    amber: "bg-amber-400/10 text-amber-400 border-amber-400/30",
    rose: "bg-rose-400/10 text-rose-400 border-rose-400/30",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function IconButton({
  children,
  label,
  onClick,
  active,
  className,
}: {
  children: ReactNode;
  label: string;
  onClick?: () => void;
  active?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className={cn(
        "inline-flex h-7 w-7 items-center justify-center rounded-md border border-transparent text-ink-400 transition-colors",
        "hover:border-[var(--color-line)] hover:bg-navy-800 hover:text-ink-100",
        active && "border-peri-400/40 bg-navy-700/60 text-peri-300",
        className,
      )}
    >
      {children}
    </button>
  );
}
