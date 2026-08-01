import { useState } from "react";
import { Ban, ChevronDown, Info, Layers } from "lucide-react";
import type { PortfolioCapability, PortfolioContextOption } from "@/domain";
import { cn } from "@/lib/utils";

/**
 * Backend-authored portfolio disclosures for the active view.
 *
 * Every sentence rendered here was written by the governed capability resolver.
 * The component chooses placement and styling; it never composes a business
 * statement, never lists portfolios it worked out itself, and never decides that
 * a scope is partial. That is the whole point: a Total workspace showing
 * originating-only pipeline data says so in the backend's words, with the
 * backend's portfolio lists, so React and Copilot cannot drift apart.
 *
 * DENSITY: the banner shows the FIRST SENTENCE of the disclosure by default;
 * the full prose and the contributing/excluded portfolio lists sit behind a
 * "Details" toggle. Nothing is removed — the governance record stays one click
 * away instead of dominating the page.
 */
export function PortfolioScopeBanner({
  context,
  capability,
  testId = "portfolio-scope-banner",
}: {
  /** The selected context, for the scope label. */
  context: PortfolioContextOption | null;
  /** The capability governing the current view. */
  capability: PortfolioCapability | undefined;
  testId?: string;
}) {
  const [open, setOpen] = useState(false);
  if (!capability) return null;
  // Nothing to disclose: fully applicable, everything in scope contributing.
  if (capability.enabled && !capability.partial && !capability.detail) return null;

  const blocked = !capability.enabled;
  const Icon = blocked ? Ban : capability.partial ? Layers : Info;

  const detail = capability.detail ?? reasonFallback(capability);
  const firstSentence = detail.match(/^.*?[.!?](?=\s|$)/)?.[0] ?? detail;
  const hasLists = capability.contributing_portfolios.length > 0;
  const hasMore = firstSentence.length < detail.length || hasLists;

  return (
    <div
      role="note"
      data-testid={testId}
      data-capability={capability.capability}
      data-enabled={capability.enabled ? "true" : "false"}
      data-partial={capability.partial ? "true" : "false"}
      className={[
        "flex items-start gap-2.5 rounded-lg border px-3 py-2 text-[12px] leading-relaxed",
        blocked
          ? "border-amber-500/30 bg-amber-500/5 text-amber-100/90"
          : "border-[var(--color-line)] bg-navy-900/50 text-ink-300",
      ].join(" ")}
    >
      <Icon size={14} className={blocked ? "mt-0.5 shrink-0 text-amber-300" : "mt-0.5 shrink-0 text-peri-300"} />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          {context && (
            <div className="text-[11px] font-medium uppercase tracking-wider text-ink-500">
              {context.label} workspace
            </div>
          )}
          {hasMore && (
            <button
              type="button"
              onClick={() => setOpen((s) => !s)}
              aria-expanded={open}
              data-testid="scope-banner-details"
              className="ml-auto inline-flex shrink-0 items-center gap-1 text-[11px] font-medium text-ink-500 hover:text-ink-300"
            >
              Details
              <ChevronDown size={13} className={cn("transition-transform", !open && "-rotate-90")} />
            </button>
          )}
        </div>
        {/* Backend prose, rendered verbatim — first sentence by default. */}
        <p className="mt-0.5">{open ? detail : firstSentence}</p>
        {open && hasLists && (
          <p className="mt-1 text-ink-400">
            Contributing: {capability.contributing_portfolios.join(", ")}
            {capability.excluded_portfolios.length > 0 && (
              <> · Excluded: {capability.excluded_portfolios.join(", ")}</>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

/** Last-resort text when the backend gave a reason code but no prose. */
function reasonFallback(capability: PortfolioCapability): string {
  return capability.reason_code
    ? `This analysis is unavailable for the selected portfolio scope (${capability.reason_code}).`
    : "This analysis is unavailable for the selected portfolio scope.";
}
