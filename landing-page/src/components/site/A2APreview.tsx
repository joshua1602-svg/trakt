/**
 * The agent-to-agent demo's text equivalent.
 *
 * Rendered only when the film cannot play — no codec, a failed load, or a
 * reduced-motion visitor who never asked for it. It carries the same recorded
 * run in words, so nobody is left with an empty frame and nobody has to watch
 * a video to learn what the section claims.
 *
 * This replaces the animated CSS loop that used to be the section's demo. The
 * loop autoplayed, had no poster and no controls, which the page's own rule
 * forbids: every demo here is user-started. The film is the demo now; this is
 * its fallback.
 *
 * Every figure is from the recorded run, not written for the page.
 */
export function A2APreview() {
  return (
    <div
      className="rounded-2xl border border-line bg-navy-900/80 p-4 shadow-[0_24px_60px_-30px_rgba(0,0,0,0.9)] sm:p-5"
      role="img"
      aria-label="The Alderbridge enterprise agent delegates a securitisation-readiness objective to a Trakt agent and receives an evidence-backed assessment"
    >
      <p className="border-b border-line pb-3 text-xs font-medium text-ink-300">
        Agent-to-agent delegation
      </p>
      <p className="pt-4 text-[13px] leading-relaxed text-ink-200">
        The Alderbridge enterprise agent discovers the Trakt Securitisation Readiness
        Agent, delegates the objective “assess this portfolio for securitisation
        readiness”, and receives a structured, evidence-backed assessment. Trakt
        chooses what to investigate.
      </p>
      <p className="mt-3 text-[13px] leading-relaxed text-ink-300">
        One recorded run against a synthetic portfolio: 30 governed queries, six
        material findings, eight diligence items and four declared gaps — including
        the 31% regional concentration that passes a warehouse limit, breaches a
        securitisation criterion and flags an internal threshold.
      </p>
    </div>
  );
}
