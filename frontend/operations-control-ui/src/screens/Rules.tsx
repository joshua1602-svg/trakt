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

function RuleRow({ rule }: { rule: Rule }) {
  const client = useOpsClient();
  const [open, setOpen] = useState(false);
  const [history, setHistory] = useState<Rule[] | null>(null);
  const [historyError, setHistoryError] = useState("");

  useEffect(() => {
    if (open && history === null) {
      client
        .getRuleHistory(rule.rule_id)
        .then(setHistory)
        .catch((err) => setHistoryError(err instanceof Error ? err.message : copy.errors.generic));
    }
  }, [open, history, client, rule.rule_id]);

  return (
    <div className="rounded-xl border border-stone-200 bg-white">
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
            <RuleRow key={rule.rule_id} rule={rule} />
          ))}
        </div>
      )}
    </Page>
  );
}
