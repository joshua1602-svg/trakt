import { useState } from "react";
import { AlertTriangle, ChevronDown, FileBarChart, RefreshCw, SlidersHorizontal, Sparkles, User } from "lucide-react";
import type { Artifact, ChatMessage as ChatMessageType, MIQuerySpec } from "@/domain";
import { ChatResult } from "@/components/ChatResult";
import { cn, formatTime, formatUiTitle } from "@/lib/utils";
import { formatPredicate } from "@/lib/filters";

export function ChatMessage({
  message,
  onOpenArtifact,
  onRetry,
  onAsk,
  onTogglePin,
  onDrill,
}: {
  message: ChatMessageType;
  onOpenArtifact?: (id: string) => void;
  onRetry?: () => void;
  /** Dispatch a suggested follow-up question (routes through context). */
  onAsk?: (question: string) => void;
  onTogglePin?: (id: string) => void;
  onDrill?: (artifact: Artifact, filters: Record<string, unknown>) => void;
}) {
  const isUser = message.role === "user";
  const hasInlineResult = !isUser && !!message.artifacts && message.artifacts.length > 0;

  return (
    <div className="animate-fade-in flex gap-2.5" data-role={message.role}>
      <div
        className={cn(
          "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg",
          isUser
            ? "bg-slate-600/40 text-slate-200"
            : message.error
              ? "bg-rose-400/15 text-rose-400"
              : "bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-sm shadow-teal-900/40",
        )}
      >
        {isUser ? <User size={14} /> : message.error ? <AlertTriangle size={14} /> : <Sparkles size={14} />}
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className={cn("text-xs font-semibold", isUser ? "text-slate-300" : "text-teal-200")}>
            {isUser ? "You" : "MI Agent"}
          </span>
          <span className="text-[10px] text-ink-500">{formatTime(message.createdAt)}</span>
        </div>

        {message.pending ? (
          <div className="mt-1.5 inline-flex items-center gap-1 rounded-lg border border-teal-700/30 bg-teal-900/20 px-3 py-2">
            <span className="dot-1 h-1.5 w-1.5 rounded-full bg-teal-300" />
            <span className="dot-2 h-1.5 w-1.5 rounded-full bg-teal-300" />
            <span className="dot-3 h-1.5 w-1.5 rounded-full bg-teal-300" />
          </div>
        ) : (
          <div
            data-testid={isUser ? "user-bubble" : "assistant-bubble"}
            className={cn(
              "mt-1 whitespace-pre-wrap rounded-2xl rounded-tl-sm border px-3.5 py-2.5 text-[13px] leading-relaxed",
              isUser
                ? "border-slate-600/30 bg-slate-700/20 text-slate-100"
                : message.error
                  ? "border-rose-400/25 bg-rose-400/5 text-rose-200"
                  : "border-teal-700/30 bg-teal-900/15 text-ink-100",
            )}
          >
            {message.content}
          </div>
        )}

        {message.error && onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-navy-800 px-2.5 py-1 text-[11px] font-medium text-ink-200 transition-colors hover:border-teal-400/40 hover:text-ink-100"
          >
            <RefreshCw size={12} />
            Retry
          </button>
        )}

        {message.assumptions && message.assumptions.length > 0 && (
          <div className="mt-2 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/50 px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-500">Assumptions</div>
            <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] leading-relaxed text-ink-400">
              {message.assumptions.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          </div>
        )}

        {message.warnings && message.warnings.length > 0 && (
          <div className="mt-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300/90">
            {message.warnings.map((w, i) => (
              <div key={i}>⚠ {w}</div>
            ))}
          </div>
        )}

        {/* The result, embedded directly in the conversation. */}
        {hasInlineResult && onTogglePin && (
          <ChatResult
            artifacts={message.artifacts!}
            onTogglePin={onTogglePin}
            onDrill={onDrill}
            onAsk={onAsk}
            onOpenArtifact={onOpenArtifact}
          />
        )}

        {!isUser && !message.pending && !message.error && <QueryLogicPanel message={message} />}

        {/* Fallback navigation links only when the result isn't embedded inline. */}
        {!hasInlineResult && message.artifactRefs && message.artifactRefs.length > 0 && (
          <div className="mt-2 flex flex-col gap-1">
            {message.artifactRefs.map((ref) => (
              <button
                key={ref.id}
                type="button"
                onClick={() => onOpenArtifact?.(ref.id)}
                className="group inline-flex items-center gap-2 rounded-md border border-[var(--color-line)] bg-navy-800/50 px-2.5 py-1.5 text-left text-[11px] text-ink-300 transition-colors hover:border-teal-400/40 hover:text-ink-100"
              >
                <FileBarChart size={13} className="text-teal-300" />
                <span className="truncate">{ref.title}</span>
                <span className="ml-auto text-[10px] uppercase tracking-wider text-ink-500 group-hover:text-teal-300">
                  {ref.type} →
                </span>
              </button>
            ))}
          </div>
        )}

        {!isUser && !message.pending && !message.error && message.suggestions && message.suggestions.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {message.suggestions.map((s) => (
              <button
                key={`${s.kind}:${s.question}`}
                type="button"
                onClick={() => onAsk?.(s.question)}
                title={s.question}
                className="inline-flex items-center rounded-full border border-teal-700/30 bg-teal-900/20 px-2.5 py-1 text-[11px] text-teal-100 transition-colors hover:border-teal-400/50 hover:bg-teal-800/30"
              >
                {s.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** One label/value row in the Query Logic panel. */
function specRows(spec?: Partial<MIQuerySpec>): Array<[string, string]> {
  if (!spec) return [];
  const rows: Array<[string, string]> = [];
  if (spec.metric) rows.push(["Measure", formatUiTitle(spec.metric)]);
  const dims = spec.dimensions?.length ? spec.dimensions : spec.dimension ? [spec.dimension] : [];
  if (dims.length) rows.push(["Dimensions", dims.map(formatUiTitle).join(", ")]);
  if (spec.aggregation) rows.push(["Aggregation", formatUiTitle(spec.aggregation)]);
  if (spec.state) rows.push(["State", spec.state]);
  if (spec.chartType) rows.push(["Chart type", formatUiTitle(spec.chartType)]);
  if (spec.riskMode) rows.push(["Risk mode", formatUiTitle(spec.riskMode)]);
  if (typeof spec.topN === "number") rows.push(["Top N", String(spec.topN)]);
  const filters = spec.filters && Object.keys(spec.filters);
  if (filters && filters.length) {
    rows.push(["Filters applied",
      filters.map((k) => formatPredicate(k, spec.filters![k])).join(" · ")]);
  }
  // Predicates the user asked for that could not be applied (never silently
  // dropped). Backend spec carries snake_case ``unavailable_filters``.
  const unavailable = (spec as { unavailable_filters?: string[] }).unavailable_filters;
  if (unavailable && unavailable.length) {
    rows.push(["Filters unavailable", unavailable.join(" · ")]);
  }
  return rows;
}

/** Coverage / source / validation rows from the message's first data artifact. */
function auditRows(message: ChatMessageType): Array<[string, string]> {
  const rows: Array<[string, string]> = [];
  const art = (message.artifacts ?? []).find(
    (a) => a.type === "chart" || a.type === "table" || a.type === "kpi");
  const recon = (art as { reconciliation?: Record<string, unknown> } | undefined)?.reconciliation;
  if (recon) {
    const inc = recon.records_included ?? recon.records_after_filters;
    const tot = recon.total_records;
    if (inc != null && tot != null) rows.push(["Rows included", `${inc} / ${tot}`]);
    const cov = recon.coverage_by_balance_pct;
    if (cov != null) rows.push(["Coverage", `${cov}% of balance`]);
  }
  if (message.warnings && message.warnings.some((w) => /not applied/i.test(w))) {
    rows.push(["Validation", "some predicates unavailable"]);
  } else if (message.spec) {
    rows.push(["Validation", "passed"]);
  }
  return rows;
}

/**
 * The single collapsed "Query Logic" disclosure for power users. Consolidates
 * the parsed interpretation, query spec, confidence and engineer diagnostics
 * that used to clutter the default chat bubble. Default view stays clean.
 */
function QueryLogicPanel({ message }: { message: ChatMessageType }) {
  const [open, setOpen] = useState(false);
  const rows = [...specRows(message.spec), ...auditRows(message)];
  const diagnostics = message.diagnostics ?? [];
  const hasContent =
    !!message.interpreted ||
    rows.length > 0 ||
    typeof message.confidence === "number" ||
    diagnostics.length > 0 ||
    message.usedContext ||
    message.cacheHit;
  if (!hasContent) return null;

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((s) => !s)}
        className="inline-flex items-center gap-1.5 text-[10px] font-medium text-ink-500 hover:text-ink-300"
      >
        <SlidersHorizontal size={11} />
        <ChevronDown size={12} className={cn("transition-transform", !open && "-rotate-90")} />
        Query logic
      </button>
      {open && (
        <div className="mt-1 space-y-2 rounded-lg border border-[var(--color-line-soft)] bg-navy-900/60 px-3 py-2.5">
          {(message.usedContext || message.cacheHit) && (
            <div className="flex flex-wrap gap-1.5">
              {message.usedContext && (
                <span className="rounded border border-peri-400/30 bg-peri-400/10 px-1.5 py-0.5 text-[10px] text-peri-200">
                  Context used{message.contextNote ? ` · ${message.contextNote}` : ""}
                </span>
              )}
              {message.cacheHit && (
                <span className="rounded border border-mint-400/30 bg-mint-400/10 px-1.5 py-0.5 text-[10px] text-mint-300">
                  Served from cache
                </span>
              )}
            </div>
          )}
          {message.interpreted && (
            <div className="font-mono text-[10px] text-ink-400">
              <span className="text-ink-500">Interpreted as · </span>
              <span>{message.interpreted}</span>
            </div>
          )}
          {rows.length > 0 && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[10px]">
              {rows.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-ink-500">{k}</dt>
                  <dd className="font-mono text-ink-300">{v}</dd>
                </div>
              ))}
              {typeof message.confidence === "number" && (
                <div className="contents">
                  <dt className="text-ink-500">Confidence</dt>
                  <dd className="font-mono text-ink-300">{Math.round(message.confidence * 100)}%</dd>
                </div>
              )}
            </dl>
          )}
          {rows.length === 0 && typeof message.confidence === "number" && (
            <div className="text-[10px] text-ink-500">
              Confidence <span className="font-mono text-ink-300">{Math.round(message.confidence * 100)}%</span>
            </div>
          )}
          {diagnostics.length > 0 && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-500">Technical details</div>
              <ul className="mt-1 list-disc space-y-0.5 pl-4 font-mono text-[10px] text-ink-400">
                {diagnostics.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
