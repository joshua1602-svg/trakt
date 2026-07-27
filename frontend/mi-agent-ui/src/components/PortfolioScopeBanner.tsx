import { Info, Layers, Ban } from "lucide-react";
import type { PortfolioCapability, PortfolioContextOption } from "@/domain";

/**
 * Backend-authored portfolio disclosures for the active view.
 *
 * Every sentence rendered here was written by the governed capability resolver.
 * The component chooses placement and styling; it never composes a business
 * statement, never lists portfolios it worked out itself, and never decides that
 * a scope is partial. That is the whole point: a Total workspace showing
 * originating-only pipeline data says so in the backend's words, with the
 * backend's portfolio lists, so React and Copilot cannot drift apart.
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
  if (!capability) return null;
  // Nothing to disclose: fully applicable, everything in scope contributing.
  if (capability.enabled && !capability.partial && !capability.detail) return null;

  const blocked = !capability.enabled;
  const Icon = blocked ? Ban : capability.partial ? Layers : Info;

  return (
    <div
      role="note"
      data-testid={testId}
      data-capability={capability.capability}
      data-enabled={capability.enabled ? "true" : "false"}
      data-partial={capability.partial ? "true" : "false"}
      className={[
        "flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-[12px] leading-relaxed",
        blocked
          ? "border-amber-500/30 bg-amber-500/5 text-amber-100/90"
          : "border-[var(--color-line)] bg-navy-900/50 text-ink-300",
      ].join(" ")}
    >
      <Icon size={14} className={blocked ? "mt-0.5 shrink-0 text-amber-300" : "mt-0.5 shrink-0 text-peri-300"} />
      <div className="min-w-0">
        {context && (
          <div className="text-[11px] font-medium uppercase tracking-wider text-ink-500">
            {context.label} workspace
          </div>
        )}
        {/* Backend prose, rendered verbatim. */}
        <p className="mt-0.5">{capability.detail ?? reasonFallback(capability)}</p>
        {capability.contributing_portfolios.length > 0 && (
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
