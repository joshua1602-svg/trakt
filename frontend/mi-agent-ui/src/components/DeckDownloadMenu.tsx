import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Check, ChevronDown, FileDown, Loader2, Presentation, Sparkles } from "lucide-react";
import type { AgentClient } from "@/api";
import type { DeckGenerationJob, DeckIndex } from "@/domain";

function formatPeriod(period?: string | null): string {
  if (!period) return "Latest";
  // "2026-01" → "Jan 2026"; pass through anything else.
  const m = /^(\d{4})-(\d{2})$/.exec(period);
  if (!m) return period;
  const d = new Date(Number(m[1]), Number(m[2]) - 1, 1);
  return d.toLocaleDateString("en-GB", { month: "short", year: "numeric" });
}

/** How often to ask the API how a generation is getting on. Generation takes
 *  seconds, so this is responsive without being a busy-wait. */
const POLL_MS = 1500;
/** Stop polling rather than hammer the API forever if a job never settles. */
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

/**
 * Investor-deck control (top-right actions).
 *
 * Two jobs in one control, because to a user they are one thing: get me the
 * pack. It discovers the decks the platform has published for the active
 * portfolio and offers them for download, AND it can ask the API to generate a
 * fresh one — which, before this, was reachable only by waiting for the nightly
 * pipeline.
 *
 * Generation is asynchronous, so the control reports the whole lifecycle rather
 * than pretending it is instant. Five outcomes are distinguished, because
 * collapsing them would misinform:
 *
 *   generating  — work in flight
 *   completed   — a new deck is published and the menu refreshes to offer it
 *   blocked     — the pack failed its publication gates. It was NOT published
 *                 and the previous deck is untouched. This is not an error.
 *   failed      — nothing was produced
 *   unavailable — this deployment (or the mock client) does not generate packs
 */
export function DeckDownloadMenu({
  client,
  portfolioId,
  reportingPeriod,
  portfolioContext,
}: {
  client: AgentClient;
  portfolioId: string;
  /** The selected funded reporting period (YYYY-MM), for the "selected date" deck. */
  reportingPeriod?: string | null;
  /** The governed workspace scope the pack should report on. */
  portfolioContext?: string | null;
}) {
  const [index, setIndex] = useState<DeckIndex | null>(null);
  const [open, setOpen] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const [job, setJob] = useState<DeckGenerationJob | null>(null);
  const [genError, setGenError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const refreshIndex = useCallback(() => {
    if (!portfolioId) return Promise.resolve();
    return client
      .getDecks(portfolioId)
      .then(setIndex)
      .catch(() => setIndex({ available: false, latest: null, decks: [], client_id: "" }));
  }, [client, portfolioId]);

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

  const inFlight = job?.state === "queued" || job?.state === "generating";

  // Poll while a generation is in flight. On completion the deck index is
  // refreshed, so the newly published pack is immediately downloadable from the
  // same menu without a page reload.
  useEffect(() => {
    if (!inFlight || !job) return;
    let cancelled = false;
    const giveUpAt = Date.now() + POLL_TIMEOUT_MS;
    const tick = () => {
      if (cancelled) return;
      if (Date.now() > giveUpAt) {
        setGenError("The pack is taking longer than expected. Try again shortly.");
        setJob(null);
        return;
      }
      const poll = client.getDeckGeneration?.bind(client);
      if (!poll) return;
      poll(job.jobId)
        .then((next) => {
          if (cancelled) return;
          setJob(next);
          if (next.state === "completed") void refreshIndex();
          if (next.state === "queued" || next.state === "generating") {
            timer = window.setTimeout(tick, POLL_MS);
          }
        })
        .catch(() => {
          if (cancelled) return;
          setGenError("Lost contact with the MI API while the pack was generating.");
          setJob(null);
        });
    };
    let timer = window.setTimeout(tick, POLL_MS);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [client, job, inFlight, refreshIndex]);

  const canServe = client.deckDownloadUrl(portfolioId) !== null;
  const available = !!index?.available && canServe;
  // A client that cannot generate decks exposes `generateDeck` as null, so the
  // control knows up front rather than discovering it when a user clicks.
  const canGenerate = !!client.generateDeck;

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

  const generate = useCallback(() => {
    setGenError(null);
    setOpen(false);
    const started = client.generateDeck?.({
      portfolioId,
      portfolioContext: portfolioContext ?? undefined,
    });
    if (!started) {
      setGenError("This environment does not generate investor packs.");
      return;
    }
    started
      .then(setJob)
      .catch((err: Error) => setGenError(err?.message || "The pack could not be requested."));
  }, [client, portfolioId, portfolioContext]);

  // Is there a deck for the selected reporting period specifically?
  const selectedPeriodDeck = reportingPeriod
    ? index?.decks.find((d) => d.period === reportingPeriod || d.period.startsWith(reportingPeriod))
    : undefined;

  // -- in flight: the control becomes a status, not an action --------------
  if (inFlight) {
    return (
      <button
        type="button"
        disabled
        data-testid="deck-download"
        title="Building the investor pack from the governed MI"
        className="inline-flex h-8 cursor-wait items-center gap-1.5 rounded-md border border-navy-600 bg-navy-800/70 px-2.5 text-[12px] font-medium text-ink-200"
      >
        <Loader2 size={14} className="animate-spin text-peri-300" />
        {job?.state === "queued" ? "Queued…" : "Generating…"}
      </button>
    );
  }

  // -- a blocked or failed generation, or a request that was refused -------
  const problem = genError
    || (job?.state === "blocked" || job?.state === "failed" ? job.message : null);
  if (problem) {
    return (
      <button
        type="button"
        onClick={() => { setJob(null); setGenError(null); }}
        data-testid="deck-generate-problem"
        title={`${problem} Click to dismiss.`}
        className="inline-flex h-8 items-center gap-1.5 rounded-md border border-amber-400/40 bg-amber-400/10 px-2.5 text-[12px] font-medium text-amber-300"
      >
        <AlertTriangle size={14} />
        {job?.state === "blocked" ? "Pack withheld" : "Pack failed"}
      </button>
    );
  }

  // -- nothing published and nothing we can do about it --------------------
  if (!available && !canGenerate) {
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

  const generateItem = canGenerate ? (
    <button
      type="button"
      role="menuitem"
      onClick={generate}
      data-testid="deck-generate"
      className="flex w-full items-center gap-2 border-t border-navy-700 px-3 py-2 text-left text-[12px] text-ink-100 transition-colors hover:bg-navy-800"
    >
      <Sparkles size={14} className="text-mint-400" />
      <span>
        Generate a new pack
        <span className="block text-[10px] text-ink-500">
          From the latest published MI · takes a few seconds
        </span>
      </span>
    </button>
  ) : null;

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
        {job?.state === "completed" && done === null
          ? <Check size={14} className="text-mint-400" />
          : <Presentation size={14} className="text-peri-300" />}
        Investor deck
        <ChevronDown size={13} className="text-ink-400" />
      </button>
      {open && (
        <div role="menu"
          className="absolute right-0 z-20 mt-1 w-60 overflow-hidden rounded-lg border border-navy-600 bg-navy-900 py-1 shadow-xl">
          {available ? (
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
          ) : (
            <div className="px-3 py-2 text-[11px] text-ink-500">
              No investor pack has been published for this portfolio yet.
            </div>
          )}

          {available && reportingPeriod && (
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

          {available && index && index.decks.length > 0 && (
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

          {generateItem}
        </div>
      )}
    </div>
  );
}
