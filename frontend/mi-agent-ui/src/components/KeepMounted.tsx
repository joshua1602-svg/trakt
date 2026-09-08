import { useRef, type ReactNode } from "react";

/**
 * Renders `children` lazily on first activation, then KEEPS THEM MOUNTED and
 * hides them with CSS when inactive.
 *
 * Why this exists: the workspace tabs were conditionally rendered
 * (`{tab === "evo" && <EvolutionPanel/>}`), so every tab change unmounted the
 * panel and destroyed its loaded data. Returning to a tab re-ran its effects
 * and re-fetched, and the panel flashed its loading state again even though
 * nothing about the data had changed.
 *
 * Two properties matter:
 *
 *  - **Lazy on first use.** A tab the user never opens is never mounted, so the
 *    initial load does not pay for secondary views. This is the "lazy load
 *    secondary content" half.
 *  - **Sticky once opened.** After the first activation the subtree stays
 *    mounted, so its state — including data already fetched — survives. This is
 *    the "retain already loaded tab data" half, and it is what makes a return
 *    visit immediate with no network at all.
 *
 * `hidden` (rather than unmount) is deliberate: it keeps the DOM out of the
 * accessibility tree and out of layout, while preserving component state.
 *
 * NOTE: this does not replace the `key` props on the panels. Those key off the
 * portfolio context / data version, so a genuine identity change still remounts
 * and refetches — which is the behaviour that must not be lost.
 */
export function KeepMounted({
  active,
  children,
  testId,
}: {
  active: boolean;
  children: ReactNode;
  testId?: string;
}) {
  // Latches on first activation and never resets.
  const everActive = useRef(false);
  if (active) everActive.current = true;
  if (!everActive.current) return null;

  return (
    // `stack-group` is the pane's internal rhythm: panels inside one tab are
    // a group, spaced closer than the sections around them.
    <div hidden={!active} data-testid={testId} aria-hidden={!active} className="stack-group">
      {children}
    </div>
  );
}
