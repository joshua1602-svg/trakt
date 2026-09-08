import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Search } from "lucide-react";
import { useOpsClient } from "@/api/context";
import type { MiQueryDetail, MiQueryFilters, MiQueryRow, MiQuerySummary } from "@/api/types";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { formatDate } from "@/lib/format";

/**
 * MI Query telemetry — the Day-1 calibration surface.
 *
 * Who asked, what they asked, what Trakt understood, which capability ran, what
 * they saw, whether it answered / refused / errored, and — after review —
 * whether the response was any good.
 *
 * Deliberately NOT the OCC system dashboard: no uptime, no runs, no gates, no
 * publications. One record type and its review, so the wider operations console
 * can later link to this rather than absorb it.
 */

const WINDOWS: { id: string; label: string }[] = [
  { id: "24h", label: "Last 24 hours" },
  { id: "72h", label: "First 72 hours" },
  { id: "7d", label: "Last 7 days" },
  { id: "all", label: "All time" },
];

const OUTCOMES = ["ANSWERED", "REFUSED", "ERROR"] as const;

const CLASSIFICATIONS = [
  "CORRECT",
  "APPROPRIATE_REFUSAL",
  "PARTIALLY_CORRECT",
  "WRONG_INTERPRETATION",
  "WRONG_CALCULATION",
  "RENDERING_ERROR",
  "SHOULD_HAVE_ANSWERED",
  "NEEDS_INVESTIGATION",
] as const;

const OUTCOME_STYLE: Record<string, string> = {
  ANSWERED: "bg-emerald-50 text-emerald-800 border-emerald-200",
  REFUSED: "bg-amber-50 text-amber-800 border-amber-200",
  ERROR: "bg-rose-50 text-rose-800 border-rose-200",
};

function humanLabel(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function Counter({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-stone-200 bg-white px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-stone-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-stone-900">{value}</div>
      {hint ? <div className="mt-0.5 text-xs text-stone-500">{hint}</div> : null}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-stone-500">{label}</div>
      <div className="mt-1 text-sm text-stone-800">{children}</div>
    </div>
  );
}

function Detail({ queryId, onReviewed }: { queryId: string; onReviewed: () => void }) {
  const client = useOpsClient();
  const [detail, setDetail] = useState<MiQueryDetail | null>(null);
  const [error, setError] = useState("");
  const [classification, setClassification] = useState<string>("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    client
      .getMiQuery(queryId)
      .then(setDetail)
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load."));
  }, [client, queryId]);

  if (error) return <ErrorNote message={error} />;
  if (!detail) return <Loading />;

  const interpretation = Object.entries(detail.interpretation ?? {});

  async function submit() {
    if (!classification) return;
    setSaving(true);
    try {
      await client.reviewMiQuery(queryId, classification, note.trim() || undefined);
      onReviewed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the review.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4 border-t border-stone-200 bg-stone-50 px-4 py-4">
      <Field label="Question">
        <p className="text-base text-stone-900">{detail.question}</p>
      </Field>

      <Field label="Answer shown to the user">
        {detail.answer ? (
          <p className="whitespace-pre-wrap text-stone-900">{detail.answer}</p>
        ) : (
          <p className="italic text-stone-500">
            {detail.message ?? "No answer was produced."}
          </p>
        )}
      </Field>

      <Field label="What Trakt understood">
        {interpretation.length ? (
          <dl className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
            {interpretation.map(([k, v]) => (
              <div key={k} className="flex gap-2">
                <dt className="text-stone-500">{humanLabel(k)}:</dt>
                <dd className="font-medium text-stone-900">
                  {typeof v === "object" ? JSON.stringify(v) : String(v)}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="italic text-stone-500">
            No structured interpretation was produced for this question.
          </p>
        )}
      </Field>

      <div className="grid gap-4 sm:grid-cols-3">
        <Field label="Capability that ran">
          {detail.route ?? detail.capability ?? "—"}
          {detail.result_type ? (
            <span className="text-stone-500"> · {detail.result_type}</span>
          ) : null}
        </Field>
        <Field label="Data version">
          {detail.snapshot_id ?? "—"}
          {detail.reporting_period ? (
            <div className="text-stone-500">Period {detail.reporting_period}</div>
          ) : null}
        </Field>
        <Field label="Outcome">
          <span
            className={`inline-block rounded-full border px-2 py-0.5 text-xs font-medium ${
              OUTCOME_STYLE[detail.outcome] ?? ""
            }`}
          >
            {detail.outcome}
          </span>
          {detail.refusal_reason ? (
            <div className="mt-1 text-stone-600">{humanLabel(detail.refusal_reason)}</div>
          ) : null}
          {detail.error_code ? (
            <div className="mt-1 text-stone-600">{humanLabel(detail.error_code)}</div>
          ) : null}
        </Field>
      </div>

      <div className="rounded-xl border border-stone-200 bg-white p-4">
        <div className="text-sm font-medium text-stone-900">Quality review</div>
        <p className="mt-1 text-xs text-stone-500">
          Calibration evidence only. This never changes the answer the client was given.
        </p>
        {detail.review_detail && detail.review_detail.classification !== "UNREVIEWED" ? (
          <p className="mt-2 text-sm text-stone-800">
            {humanLabel(detail.review_detail.classification)} — reviewed by{" "}
            {detail.review_detail.reviewer} on{" "}
            {formatDate(detail.review_detail.reviewed_at ?? "")}
            {detail.review_detail.note ? `: ${detail.review_detail.note}` : ""}
          </p>
        ) : null}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <select
            value={classification}
            onChange={(e) => setClassification(e.target.value)}
            className="rounded-lg border border-stone-300 px-3 py-1.5 text-sm"
            aria-label="Quality classification"
          >
            <option value="">Choose a classification…</option>
            {CLASSIFICATIONS.map((c) => (
              <option key={c} value={c}>
                {humanLabel(c)}
              </option>
            ))}
          </select>
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Note (optional)"
            className="min-w-[16rem] flex-1 rounded-lg border border-stone-300 px-3 py-1.5 text-sm"
            aria-label="Review note"
          />
          <button
            type="button"
            onClick={submit}
            disabled={!classification || saving}
            className="rounded-lg bg-stone-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
          >
            {saving ? "Saving…" : "Save review"}
          </button>
        </div>
      </div>
    </div>
  );
}

export function MiQueriesScreen() {
  const client = useOpsClient();
  const [filters, setFilters] = useState<MiQueryFilters>({ window: "72h" });
  const [summary, setSummary] = useState<MiQuerySummary | null>(null);
  const [rows, setRows] = useState<MiQueryRow[] | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const load = useCallback(() => {
    setError("");
    Promise.all([client.getMiQuerySummary(filters), client.getMiQueries(filters)])
      .then(([s, r]) => {
        setSummary(s);
        setRows(r);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load."));
  }, [client, filters]);

  useEffect(load, [load]);

  const visible = useMemo(() => {
    if (!rows) return null;
    const needle = search.trim().toLowerCase();
    return needle ? rows.filter((r) => r.question.toLowerCase().includes(needle)) : rows;
  }, [rows, search]);

  return (
    <Page
      title="MI Query usage"
      subtitle="What real users asked, what Trakt understood, and whether the response was good."
    >
      <div className="flex flex-wrap items-center gap-2">
        {WINDOWS.map((w) => (
          <button
            key={w.id}
            type="button"
            onClick={() => setFilters((f) => ({ ...f, window: w.id }))}
            className={`rounded-lg border px-3 py-1.5 text-sm ${
              filters.window === w.id
                ? "border-stone-900 bg-stone-900 text-white"
                : "border-stone-300 bg-white text-stone-700"
            }`}
          >
            {w.label}
          </button>
        ))}
      </div>

      {error ? <ErrorNote message={error} /> : null}

      {summary ? (
        <>
          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
            <Counter label="Questions" value={String(summary.total_questions)} />
            <Counter label="Users" value={String(summary.unique_users)} />
            <Counter
              label="Answered"
              value={String(summary.answered)}
              hint={summary.answered_pct !== null ? `${summary.answered_pct}%` : undefined}
            />
            <Counter
              label="Refused"
              value={String(summary.refused)}
              hint={summary.refused_pct !== null ? `${summary.refused_pct}%` : undefined}
            />
            <Counter
              label="Errors"
              value={String(summary.errors)}
              hint={summary.error_pct !== null ? `${summary.error_pct}%` : undefined}
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Counter label="Unreviewed" value={String(summary.unreviewed)} />
            <Counter
              label="Reviewed"
              value={String(summary.reviewed)}
              hint={`${summary.reviewed_correct} correct · ${summary.reviewed_problematic} problematic`}
            />
            {/* The rate NEVER travels without its denominator: it is a rate over
                the reviewed subset, not an accuracy figure for the agent. */}
            <Counter
              label="Reviewed correctness"
              value={
                summary.reviewed_correctness_pct !== null
                  ? `${summary.reviewed_correctness_pct}%`
                  : "—"
              }
              hint={`of ${summary.reviewed} reviewed`}
            />
            <Counter
              label="Latency"
              value={summary.median_latency_ms !== null ? `${summary.median_latency_ms} ms` : "—"}
              hint={summary.p95_latency_ms !== null ? `p95 ${summary.p95_latency_ms} ms` : undefined}
            />
          </div>
        </>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-stone-400" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search questions"
            className="rounded-lg border border-stone-300 py-1.5 pl-9 pr-3 text-sm"
            aria-label="Search questions"
          />
        </div>
        <select
          value={filters.outcome ?? ""}
          onChange={(e) => setFilters((f) => ({ ...f, outcome: e.target.value || undefined }))}
          className="rounded-lg border border-stone-300 px-3 py-1.5 text-sm"
          aria-label="Filter by outcome"
        >
          <option value="">All outcomes</option>
          {OUTCOMES.map((o) => (
            <option key={o} value={o}>
              {humanLabel(o)}
            </option>
          ))}
        </select>
        <select
          value={filters.review ?? ""}
          onChange={(e) => setFilters((f) => ({ ...f, review: e.target.value || undefined }))}
          className="rounded-lg border border-stone-300 px-3 py-1.5 text-sm"
          aria-label="Filter by review"
        >
          <option value="">Any review state</option>
          <option value="UNREVIEWED">Unreviewed</option>
          <option value="PROBLEMATIC">Problematic</option>
          {CLASSIFICATIONS.map((c) => (
            <option key={c} value={c}>
              {humanLabel(c)}
            </option>
          ))}
        </select>
      </div>

      {visible === null ? (
        <Loading />
      ) : visible.length === 0 ? (
        <p className="rounded-xl border border-stone-200 bg-white px-4 py-6 text-sm text-stone-600">
          No questions in this period.
        </p>
      ) : (
        <div className="space-y-2">
          {visible.map((row) => (
            <div key={row.query_id} className="rounded-xl border border-stone-200 bg-white">
              <button
                type="button"
                onClick={() => setOpen((o) => (o === row.query_id ? null : row.query_id))}
                className="flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-stone-50"
              >
                {open === row.query_id ? (
                  <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-stone-400" aria-hidden />
                ) : (
                  <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-stone-400" aria-hidden />
                )}
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-stone-900">{row.question}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-stone-500">
                    <span>{formatDate(row.asked_at)}</span>
                    <span>{row.user_id ?? "—"}</span>
                    {row.route ? <span>{row.route}</span> : null}
                    {row.duration_ms !== null ? <span>{row.duration_ms} ms</span> : null}
                    {row.review !== "UNREVIEWED" ? (
                      <span className="font-medium text-stone-700">{humanLabel(row.review)}</span>
                    ) : null}
                  </div>
                </div>
                <span
                  className={`shrink-0 rounded-full border px-2 py-0.5 text-xs font-medium ${
                    OUTCOME_STYLE[row.outcome] ?? ""
                  }`}
                >
                  {row.outcome}
                </span>
              </button>
              {open === row.query_id ? (
                <Detail queryId={row.query_id} onReviewed={load} />
              ) : null}
            </div>
          ))}
        </div>
      )}
    </Page>
  );
}
