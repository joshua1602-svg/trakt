import type { ReactNode } from "react";
import { FlaskConical } from "lucide-react";
import { copy } from "@/lib/copy";

/**
 * Pieces shared by the two OCC Agent screens.
 *
 * Nothing here decides anything. `stateTone` maps a case state onto the chip
 * palette the rest of the OCC already uses, so a practice case reads the same
 * way a live one does; the state itself always comes from the backend.
 */

/** Map a case state onto the existing OCC chip vocabulary. */
export function stateTone(state: string): string {
  if (state === "READY_FOR_EXECUTION") return "ready";
  if (state === "BLOCKED") return "blocked";
  if (state === "CANCELLED") return "cancelled";
  if (state === "SYNTHETIC_ONBOARDING_RUNNING") return "running";
  if (state === "SYNTHETIC_ONBOARDING_PASSED") return "completed";
  if (/REQUIRED|EXCEPTIONS/.test(state)) return "needs_review";
  return "waiting";
}

/** The banner every OCC Agent screen leads with. */
export function SyntheticBanner() {
  return (
    <div
      role="note"
      className="flex items-start gap-3 rounded-2xl border border-violet-200 bg-violet-50 px-4 py-3 text-sm text-violet-900"
    >
      <FlaskConical className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <p>{copy.agent.syntheticBanner}</p>
    </div>
  );
}

/** A titled block, matching the card rhythm used across the OCC. */
export function Panel({
  title,
  action,
  children,
}: {
  title: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-stone-900">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

/** A label/value row. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-stone-100 py-1.5 last:border-0">
      <span className="text-xs uppercase tracking-wide text-stone-400">{label}</span>
      <span className="text-sm text-stone-800">{children}</span>
    </div>
  );
}

export function Empty({ text = copy.agent.nothingYet }: { text?: string }) {
  return <p className="text-sm text-stone-400">{text}</p>;
}
