import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Search } from "lucide-react";
import { useOpsClient } from "@/api/context";
import type { Rule } from "@/api/types";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { copy } from "@/lib/copy";
import { formatDate, humanize } from "@/lib/format";
import { useLoad } from "@/lib/useLoad";

const SCOPE_LABELS: Record<string, string> = {
  file: copy.scopes.file,
  portfolio: copy.scopes.portfolio,
  client: copy.scopes.client,
  global: copy.scopes.global,
};

function RuleRow({ rule, onRetired }: { rule: Rule; onRetired: () => void }) {
  const client = useOpsClient();
  const [open, setOpen] = useState(false);
  const [history, setHistory] = useState<Rule[] | null>(null);
  const [historyError, setHistoryError] = useState("");
  /** Withdrawing is behind a reason box, not a bare button: this rule is read
   *  on every future delivery, and what is typed here is the only thing that
   *  explains the change months from now. */
  const [withdrawing, setWithdrawing] = useState(false);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function retire() {
    setBusy(true);
    setError("");
    try {
      await client.retireRule(rule.rule_id, reason, rule.client_id || undefined);
      setWithdrawing(false);
      setReason("");
      onRetired();
    } catch (err) {
      setError(err instanceof Error ? err.message : copy.errors.generic);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (open && history === null) {
      client
        .getRuleHistory(rule.rule_id)
        .then(setHistory)
        .catch((err) => setHistoryError(err instanceof Error ? err.message : copy.errors.generic));
    }
  }, [open, history, client, rule.rule_id]);

  return (
    <div data-rule={rule.rule_id} role="group" aria-label={rule.source_term}
         className="rounded-xl border border-stone-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-stone-50"
      >
        {open ? (
          <ChevronDown className="h-4 w-4 shrink-0 text-stone-400" aria-hidden />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0 text-stone-400" aria-hidden />
        )}
        <div className="grid min-w-0 flex-1 gap-1 sm:grid-cols-2">
          <div className="min-w-0">
            <p className="text-xs text-stone-400">{copy.rules.sourceTerm}</p>
            <p className="truncate text-sm font-medium text-stone-900">{rule.source_term}</p>
          </div>
          <div className="min-w-0">
            <p className="text-xs text-stone-400">{copy.rules.approvedMeaning}</p>
            <p className="truncate text-sm text-stone-800">{rule.approved_meaning}</p>
          </div>
        </div>
        <div className="hidden shrink-0 items-center gap-2 text-xs text-stone-500 md:flex">
          <span className="rounded-full bg-stone-100 px-2.5 py-0.5 font-medium text-stone-600">
            {SCOPE_LABELS[rule.scope] ?? humanize(rule.scope)}
          </span>
          {rule.client_id && <span>{rule.client_id}</span>}
          <span>{rule.approved_by}</span>
          <span>{formatDate(rule.approved_at)}</span>
          <span className="font-medium text-stone-400">v{rule.version}</span>
        </div>
      </button>
      {open && (
        <div className="border-t border-stone-100 px-11 py-4">
          <p className="mb-4 text-sm leading-relaxed text-stone-600">{rule.description}</p>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-400">
            {copy.rules.historyHeading}
          </h4>
          {historyError && <p className="text-sm text-rose-700">{historyError}</p>}
          {history === null && !historyError && (
            <p className="text-sm text-stone-400">{copy.common.loading}</p>
          )}
          {history && (
            <ul className="space-y-1.5">
              {history.map((version) => (
                <li key={version.version} className="flex flex-wrap items-center gap-3 text-sm">
                  <span className="w-8 font-medium text-stone-400">v{version.version}</span>
                  <span className="text-stone-800">{version.approved_meaning}</span>
                  <span className="text-xs text-stone-400">
                    {version.approved_by} · {formatDate(version.approved_at)}
                  </span>
                </li>
              ))}
            </ul>
          )}

          {rule.status === "active" && (
            <div className="mt-5 border-t border-stone-100 pt-4">
              {!withdrawing ? (
                <button
                  type="button"
                  onClick={() => setWithdrawing(true)}
                  className="text-sm font-medium text-rose-700 hover:text-rose-800"
                >
                  {copy.rules.retire}
                </button>
              ) : (
                <div role="group" aria-label={copy.rules.retireHeading}>
                  <p className="text-sm text-stone-700">{copy.rules.retireHelp}</p>
                  <label
                    htmlFor={`retire-${rule.rule_id}`}
                    className="mt-3 block text-xs text-stone-500"
                  >
                    {copy.rules.retireReason}
                  </label>
                  <textarea
                    id={`retire-${rule.rule_id}`}
                    rows={2}
                    value={reason}
                    onChange={(event) => setReason(event.target.value)}
                    className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
                  />
                  <p className="mt-1 text-xs text-stone-400">{copy.rules.retireReasonHelp}</p>
                  {error && <p className="mt-2 text-sm text-rose-700">{error}</p>}
                  <div className="mt-3 flex gap-2">
                    <button
                      type="button"
                      disabled={busy || !reason.trim()}
                      onClick={() => void retire()}
                      className="rounded-xl bg-rose-700 px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
                    >
                      {copy.rules.retireConfirm}
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => {
                        setWithdrawing(false);
                        setError("");
                      }}
                      className="rounded-xl border border-stone-300 px-4 py-2 text-sm font-semibold text-stone-700"
                    >
                      {copy.rules.retireCancel}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function RulesScreen() {
  const client = useOpsClient();
  const [q, setQ] = useState("");
  const [kind, setKind] = useState("");
  const [scope, setScope] = useState("");
  const [kinds, setKinds] = useState<string[]>([]);

  const { data, error, loading, reload } = useLoad(
    () =>
      client.getRules({
        q: q || undefined,
        kind: kind || undefined,
        scope: scope || undefined,
      }),
    [q, kind, scope],
  );

  // Collect the kinds we've seen so the filter offers real values.
  useEffect(() => {
    if (data) {
      setKinds((prev) => {
        const merged = new Set([...prev, ...data.map((r) => r.kind)]);
        return [...merged].sort();
      });
    }
  }, [data]);

  return (
    <Page title={copy.rules.title} subtitle={copy.rules.subtitle}>
      <div className="mb-6 flex flex-wrap gap-3">
        <div className="relative min-w-56 flex-1">
          <Search
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-stone-400"
            aria-hidden
          />
          <input
            value={q}
            onChange={(event) => setQ(event.target.value)}
            placeholder={copy.rules.searchPlaceholder}
            className="w-full rounded-xl border border-stone-300 bg-white py-2.5 pl-9 pr-3 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        </div>
        <select
          value={kind}
          onChange={(event) => setKind(event.target.value)}
          className="rounded-xl border border-stone-300 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-500"
        >
          <option value="">{copy.rules.kindAll}</option>
          {kinds.map((value) => (
            <option key={value} value={value}>
              {humanize(value)}
            </option>
          ))}
        </select>
        <select
          value={scope}
          onChange={(event) => setScope(event.target.value)}
          className="rounded-xl border border-stone-300 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-500"
        >
          <option value="">{copy.rules.scopeAll}</option>
          {Object.entries(SCOPE_LABELS).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>

      {loading && <Loading />}
      {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
      {data && !loading && (
        <div className="space-y-2">
          {data.length === 0 && (
            <p className="rounded-xl border border-dashed border-stone-200 px-4 py-10 text-center text-sm text-stone-400">
              {copy.rules.empty}
            </p>
          )}
          {data.map((rule) => (
            <RuleRow key={rule.rule_id} rule={rule}
                     onRetired={() => void reload()} />
          ))}
        </div>
      )}
    </Page>
  );
}
