import { useCallback, useEffect, useRef, useState } from "react";
import { Check, ChevronDown, FileDown, Presentation } from "lucide-react";
import type { AgentClient } from "@/api";
import type { DeckIndex } from "@/domain";

function formatPeriod(period?: string | null): string {
  if (!period) return "Latest";
  // "2026-01" → "Jan 2026"; pass through anything else.
  const m = /^(\d{4})-(\d{2})$/.exec(period);
  if (!m) return period;
  const d = new Date(Number(m[1]), Number(m[2]) - 1, 1);
  return d.toLocaleDateString("en-GB", { month: "short", year: "numeric" });
}

/**
 * Investor-deck download control (top-right actions). Discovers the PPTX decks
 * the orchestration published for the active portfolio and offers the latest
 * deck plus any dated reporting-period decks — never a raw blob path. When no
 * deck exists (or the client can't serve decks, e.g. mock) the control is
 * disabled with a clear "No deck available" state.
 */
export function DeckDownloadMenu({
  client,
  portfolioId,
  reportingPeriod,
}: {
  client: AgentClient;
  portfolioId: string;
  /** The selected funded reporting period (YYYY-MM), for the "selected date" deck. */
  reportingPeriod?: string | null;
}) {
  const [index, setIndex] = useState<DeckIndex | null>(null);
  const [open, setOpen] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!portfolioId) return;
    let cancelled = false;
    client
      .getDecks(portfolioId)
      .then((idx) => { if (!cancelled) setIndex(idx); })
      .catch(() => { if (!cancelled) setIndex({ available: false, latest: null, decks: [], client_id: "" }); });
    return () => { cancelled = true; };
  }, [client, portfolioId]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const canServe = client.deckDownloadUrl(portfolioId) !== null;
  const available = !!index?.available && canServe;

  const download = useCallback((period: string | null, key: string) => {
    const url = client.deckDownloadUrl(portfolioId, period);
    if (!url) return;
    // Navigate to the download endpoint (Content-Disposition: attachment).
    const a = document.createElement("a");
    a.href = url;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setDone(key);
    setTimeout(() => setDone((d) => (d === key ? null : d)), 1400);
    setOpen(false);
  }, [client, portfolioId]);

  // Is there a deck for the selected reporting period specifically?
  const selectedPeriodDeck = reportingPeriod
    ? index?.decks.find((d) => d.period === reportingPeriod || d.period.startsWith(reportingPeriod))
    : undefined;

  if (!available) {
    return (
      <button
        type="button"
        disabled
        data-testid="deck-download"
        title={canServe ? "No investor deck available for this portfolio yet"
          : "Investor decks are served by the live MI API"}
        className="inline-flex h-8 cursor-not-allowed items-center gap-1.5 rounded-md border border-navy-600 bg-navy-900/40 px-2.5 text-[12px] font-medium text-ink-500 opacity-70"
      >
        <Presentation size={14} /> No deck
      </button>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        data-testid="deck-download"
        aria-haspopup="menu"
        aria-expanded={open}
        className="inline-flex h-8 items-center gap-1.5 rounded-md border border-navy-600 bg-navy-800/70 px-2.5 text-[12px] font-medium text-ink-100 ring-1 ring-inset ring-white/5 transition-colors hover:bg-navy-700"
      >
        <Presentation size={14} className="text-peri-300" /> Investor deck
        <ChevronDown size={13} className="text-ink-400" />
      </button>
      {open && (
        <div role="menu"
          className="absolute right-0 z-20 mt-1 w-60 overflow-hidden rounded-lg border border-navy-600 bg-navy-900 py-1 shadow-xl">
          <button
            type="button"
            role="menuitem"
            onClick={() => download(null, "latest")}
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-[12px] text-ink-100 transition-colors hover:bg-navy-800"
          >
            {done === "latest" ? <Check size={14} className="text-mint-400" /> : <FileDown size={14} className="text-peri-300" />}
            <span>
              Latest deck
              {index?.latest?.period && (
                <span className="block text-[10px] text-ink-500">{formatPeriod(index.latest.period)}</span>
              )}
            </span>
          </button>

          {reportingPeriod && (
            <button
              type="button"
              role="menuitem"
              disabled={!selectedPeriodDeck}
              onClick={() => selectedPeriodDeck && download(selectedPeriodDeck.period, "selected")}
              title={selectedPeriodDeck ? undefined : "No deck available for this reporting date"}
              className={selectedPeriodDeck
                ? "flex w-full items-center gap-2 px-3 py-2 text-left text-[12px] text-ink-100 transition-colors hover:bg-navy-800"
                : "flex w-full cursor-not-allowed items-center gap-2 px-3 py-2 text-left text-[12px] text-ink-500"}
            >
              {done === "selected" ? <Check size={14} className="text-mint-400" /> : <FileDown size={14} className="text-peri-300" />}
              <span>
                Selected reporting date
                <span className="block text-[10px] text-ink-500">
                  {selectedPeriodDeck ? formatPeriod(reportingPeriod) : "No deck available for this reporting date"}
                </span>
              </span>
            </button>
          )}

          {index && index.decks.length > 0 && (
            <>
              <div className="mt-1 border-t border-navy-700 px-3 pb-1 pt-2 text-[10px] uppercase tracking-wider text-ink-500">
                By reporting period
              </div>
              {index.decks.map((d) => (
                <button
                  key={d.period}
                  type="button"
                  role="menuitem"
                  onClick={() => download(d.period, d.period)}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[12px] text-ink-200 transition-colors hover:bg-navy-800"
                >
                  {done === d.period ? <Check size={13} className="text-mint-400" /> : <FileDown size={13} className="text-ink-400" />}
                  {formatPeriod(d.period)}
                </button>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
