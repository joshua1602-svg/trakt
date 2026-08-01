/**
 * Concentration Tests — the operator review surface over /ops/concentration/*.
 *
 * The screen renders what the governed backend decided and forwards operator
 * decisions; it never calculates a metric, never derives an outcome, and
 * never lets a browser check stand in for the server's authorisation. The
 * approve button is disabled while blockers remain and says exactly why —
 * mirroring (not replacing) the server's refusal, which is still surfaced
 * verbatim if it disagrees.
 */

import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import { useSession } from "@/api/session";
import type {
  ConcentrationReviewPackage,
  ProposalStatus,
  TestProposal,
} from "@/api/concentrationTypes";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { Page } from "@/components/Page";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import {
  Card,
  Field,
  PrimaryButton,
  SecondaryButton,
  SectionHeading,
  TextInput,
} from "@/components/onboarding/primitives";
import { copy } from "@/lib/copy";
import { formatDate } from "@/lib/format";
import { errorMessage, useLoad } from "@/lib/useLoad";

const OUTCOME_TONE: Record<string, string> = {
  matched: "bg-emerald-50 text-emerald-700 border-emerald-200",
  composed: "bg-blue-50 text-blue-700 border-blue-200",
  ambiguous: "bg-amber-50 text-amber-700 border-amber-200",
  unsupported: "bg-stone-100 text-stone-600 border-stone-200",
};

const FILTERS: (ProposalStatus | "all")[] = [
  "all", "pending_confirmation", "pending_approval", "approved", "rejected",
  "unsupported", "not_applicable", "clarification_requested", "superseded",
];

/** Why approval is blocked, in the backend's own vocabulary (display only —
 *  the server refuses independently; this just explains the disabled button). */
function approvalBlockers(p: TestProposal): string[] {
  const blockers: string[] = [];
  if (p.match_outcome === "unsupported" || p.status === "unsupported") {
    blockers.push("This test has no supported implementation.");
  }
  if (p.concern_codes.length > 0) {
    blockers.push(`Unresolved concerns: ${p.concern_codes.join(", ")}.`);
  }
  if (p.confirmation_questions.length > 0) {
    blockers.push("Unanswered confirmation questions remain.");
  }
  if (p.extracted_threshold === null) blockers.push("No threshold is set.");
  if (p.extracted_operator !== "max" && p.extracted_operator !== "min") {
    blockers.push("The comparison direction is unresolved.");
  }
  if (!["pending_approval", "pending_confirmation"].includes(p.status)) {
    blockers.push(`A proposal in status '${p.status}' cannot be approved.`);
  }
  return blockers;
}

export function ConcentrationClientsScreen() {
  const client = useOpsClient();
  const navigate = useNavigate();
  const [chosen, setChosen] = useState("");
  const clients = useLoad(() => client.getClients(), []);
  return (
    <Page title={copy.concentration.title} subtitle={copy.concentration.subtitle}>
      {clients.loading && <Loading />}
      {clients.error && !clients.loading && (
        <ErrorNote message={clients.error} onRetry={() => void clients.reload()} />
      )}
      {clients.data && !clients.loading && (
        <Card>
          <p className="text-sm text-stone-600">{copy.concentration.pickClient}</p>
          <div className="mt-3 flex items-end gap-3">
            <label className="block grow">
              <span className="text-xs uppercase tracking-wide text-stone-400">
                Client
              </span>
              <select
                value={chosen}
                onChange={(e) => setChosen(e.target.value)}
                className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm"
              >
                <option value="">—</option>
                {clients.data.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <PrimaryButton
              disabled={!chosen}
              onClick={() => navigate(`/concentration/${encodeURIComponent(chosen)}`)}
            >
              {copy.concentration.open}
            </PrimaryButton>
          </div>
        </Card>
      )}
    </Page>
  );
}

export function ConcentrationReviewScreen() {
  const { client: clientId = "" } = useParams();
  const ops = useOpsClient();
  const toast = useToast();
  const { isAdmin } = useSession();
  const [busy, setBusy] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("all");
  const [sourceRef, setSourceRef] = useState("client covenant response");
  const [responseText, setResponseText] = useState("");
  const [extracted, setExtracted] = useState(false);

  const view = useLoad<ConcentrationReviewPackage>(
    () => ops.getConcentrationReview(clientId), [clientId]);

  async function act<T>(run: () => Promise<T>): Promise<T | undefined> {
    if (busy) return undefined;
    setBusy(true);
    try {
      const result = await run();
      await view.reload({ quiet: true });
      return result;
    } catch (err) {
      toast.show(errorMessage(err), "error");
      return undefined;
    } finally {
      setBusy(false);
    }
  }

  const pkg = view.data;
  const proposals = useMemo(() => pkg?.proposals ?? [], [pkg]);
  const filtered = filter === "all"
    ? proposals
    : proposals.filter((p) => p.status === filter);
  const selected = proposals.find((p) => p.proposal_id === selectedId) ?? null;
  const approved = proposals.filter((p) => p.status === "approved");

  if (view.loading && !pkg) return <Loading />;
  if (view.error && !pkg) {
    return (
      <Page title={copy.concentration.title}>
        <ErrorNote message={view.error} onRetry={() => void view.reload()} />
      </Page>
    );
  }
  if (!pkg) return null;

  return (
    <Page
      title={`${copy.concentration.title} — ${clientId}`}
      subtitle={copy.concentration.subtitle}
      actions={
        <Link
          to="/concentration"
          className="inline-flex items-center gap-1 text-sm text-stone-600 hover:text-stone-900"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden /> All clients
        </Link>
      }
    >
      {!isAdmin && (
        <p className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          {copy.concentration.permissionNote}
        </p>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="space-y-6">
          {/* Client response → extraction */}
          <Card>
            <SectionHeading>{copy.concentration.extractHeading}</SectionHeading>
            <p className="mt-1 text-sm text-stone-600">{copy.concentration.extractHelp}</p>
            <div className="mt-3 space-y-3">
              <Field label={copy.concentration.sourceReference}>
                <TextInput value={sourceRef} onChange={setSourceRef} />
              </Field>
              <label className="block">
                <span className="text-xs uppercase tracking-wide text-stone-400">
                  {copy.concentration.responseText}
                </span>
                <textarea
                  value={responseText}
                  onChange={(e) => setResponseText(e.target.value)}
                  rows={4}
                  className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm"
                />
              </label>
              <PrimaryButton
                disabled={busy || !responseText.trim()}
                onClick={() =>
                  void act(async () => {
                    const out = await ops.extractConcentrationTests(
                      clientId, sourceRef, responseText);
                    setExtracted(true);
                    if (out.count === 0) {
                      toast.show(copy.concentration.noneFound, "info");
                    } else {
                      toast.show(`${out.count} proposal(s) read from the response.`, "success");
                    }
                  })
                }
              >
                {copy.concentration.extract}
              </PrimaryButton>
            </div>
          </Card>

          {/* Proposals */}
          <Card>
            <div className="flex items-center justify-between gap-3">
              <SectionHeading>{copy.concentration.proposalsHeading}</SectionHeading>
              <select
                aria-label="Filter proposals"
                value={filter}
                onChange={(e) => setFilter(e.target.value as typeof filter)}
                className="rounded-xl border border-stone-300 px-2 py-1.5 text-sm"
              >
                {FILTERS.map((f) => (
                  <option key={f} value={f}>
                    {f === "all"
                      ? copy.concentration.filterAll
                      : (copy.statusLabels as Record<string, string>)[f] ?? f}
                  </option>
                ))}
              </select>
            </div>
            {proposals.length === 0 && (
              <p className="mt-3 rounded-xl border border-dashed border-stone-200 px-4 py-8 text-center text-sm text-stone-400">
                {extracted ? copy.concentration.noneFound : copy.concentration.noResponse}
              </p>
            )}
            <div className="mt-3 space-y-2">
              {filtered.map((p) => (
                <button
                  key={p.proposal_id}
                  type="button"
                  onClick={() =>
                    setSelectedId(selectedId === p.proposal_id ? null : p.proposal_id)
                  }
                  aria-expanded={selectedId === p.proposal_id}
                  className={clsx(
                    "flex w-full items-center justify-between gap-3 rounded-xl border px-4 py-3 text-left transition-colors",
                    selectedId === p.proposal_id
                      ? "border-stone-400 bg-stone-50"
                      : "border-stone-200 bg-white hover:border-stone-300",
                  )}
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium text-stone-900">
                      {p.extracted_test_name || p.proposed_metric_id || "Unnamed test"}
                    </span>
                    <span className="block text-xs text-stone-500">
                      {p.proposed_metric_id || "no measure"} ·{" "}
                      {p.extracted_operator || "?"}{" "}
                      {p.extracted_threshold ?? "?"}
                      {p.extracted_unit === "percent" ? "%" : ""}
                    </span>
                  </span>
                  <span className="flex shrink-0 items-center gap-2">
                    <span
                      className={clsx(
                        "rounded-full border px-2.5 py-0.5 text-xs font-medium",
                        OUTCOME_TONE[p.match_outcome],
                      )}
                    >
                      {p.match_outcome}
                    </span>
                    <StatusChip status={p.status} />
                  </span>
                </button>
              ))}
            </div>
          </Card>

          {/* Detail */}
          <Card>
            <SectionHeading>{copy.concentration.detailHeading}</SectionHeading>
            {!selected ? (
              <p className="mt-3 text-sm text-stone-400">
                {copy.concentration.selectPrompt}
              </p>
            ) : (
              <ProposalDetail
                key={`${selected.proposal_id}-${selected.version}`}
                proposal={selected}
                clientId={clientId}
                busy={busy}
                isAdmin={isAdmin}
                act={act}
              />
            )}
          </Card>
        </div>

        {/* Right rail: status, activation, versions */}
        <aside className="space-y-6">
          <Card>
            <SectionHeading>Status</SectionHeading>
            <dl className="mt-2 space-y-1 text-sm">
              {Object.entries(pkg.countsByStatus).map(([status, n]) => (
                <div key={status} className="flex items-center justify-between">
                  <dt>
                    <StatusChip status={status} />
                  </dt>
                  <dd className="font-medium text-stone-700">{n}</dd>
                </div>
              ))}
              {Object.keys(pkg.countsByStatus).length === 0 && (
                <p className="text-sm text-stone-400">No proposals yet.</p>
              )}
            </dl>
          </Card>

          <Card>
            <SectionHeading>{copy.concentration.activationHeading}</SectionHeading>
            <p className="mt-1 text-sm text-stone-600">
              {copy.concentration.activationIntro}
            </p>
            {pkg.currentVersion !== null && (
              <p className="mt-2 text-sm text-stone-700">
                {copy.concentration.currentVersion}: v{pkg.currentVersion} (
                {pkg.currentTestCount} test(s))
              </p>
            )}
            {approved.length > 0 ? (
              <>
                <p className="mt-3 text-xs uppercase tracking-wide text-stone-400">
                  {copy.concentration.willActivate}
                </p>
                <ul className="mt-1 list-inside list-disc text-sm text-stone-700">
                  {approved.map((p) => (
                    <li key={p.proposal_id}>
                      {p.extracted_test_name} ({p.extracted_operator}{" "}
                      {p.extracted_threshold}
                      {p.extracted_unit === "percent" ? "%" : ""})
                    </li>
                  ))}
                </ul>
                <div className="mt-3">
                <PrimaryButton
                  disabled={busy || !isAdmin}
                  onClick={() =>
                    void act(async () => {
                      const config = await ops.activateConcentrationTests(clientId);
                      toast.show(
                        `Version ${config.version} is now active (${config.tests.length} test(s)).`,
                        "success",
                      );
                    })
                  }
                >
                  {copy.concentration.activate}
                </PrimaryButton>
                </div>
              </>
            ) : (
              <p className="mt-3 text-sm text-stone-400">
                {copy.concentration.nothingToActivate}
              </p>
            )}
          </Card>

          <Card>
            <SectionHeading>{copy.concentration.versionsHeading}</SectionHeading>
            {pkg.versions.length === 0 ? (
              <p className="mt-2 text-sm text-stone-400">No versions yet.</p>
            ) : (
              <ul className="mt-2 space-y-2 text-sm">
                {pkg.versions.map((v) => (
                  <li
                    key={v.version}
                    className="flex items-center justify-between rounded-xl border border-stone-200 px-3 py-2"
                  >
                    <span>
                      v{v.version} · {v.test_count} test(s)
                      <span className="block text-xs text-stone-500">
                        {v.activated_by} · {formatDate(v.activated_at)}
                      </span>
                    </span>
                    <StatusChip status={v.status} />
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </aside>
      </div>
    </Page>
  );
}

function ProposalDetail({
  proposal: p,
  clientId,
  busy,
  isAdmin,
  act,
}: {
  proposal: TestProposal;
  clientId: string;
  busy: boolean;
  isAdmin: boolean;
  act: <T>(run: () => Promise<T>) => Promise<T | undefined>;
}) {
  const ops = useOpsClient();
  const toast = useToast();
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [paramsJson, setParamsJson] = useState(
    JSON.stringify(p.parameters, null, 2),
  );
  const [threshold, setThreshold] = useState(
    p.extracted_threshold === null ? "" : String(p.extracted_threshold),
  );
  const [operator, setOperator] = useState(p.extracted_operator);
  const [effectiveDate, setEffectiveDate] = useState(p.effective_date);
  const [resolved, setResolved] = useState<Record<string, boolean>>({});
  const [approveComments, setApproveComments] = useState("");
  const [reason, setReason] = useState("");
  const [clarifyQ, setClarifyQ] = useState("");

  const blockers = approvalBlockers(p);
  const decided = ["approved", "rejected", "superseded", "not_applicable",
                   "unsupported"].includes(p.status);

  function saveEdits() {
    let parameters: Record<string, unknown> | undefined;
    try {
      parameters = JSON.parse(paramsJson) as Record<string, unknown>;
    } catch {
      toast.show(copy.concentration.invalidJson, "error");
      return;
    }
    void act(() =>
      ops.updateConcentrationProposal(clientId, p.proposal_id, {
        parameters,
        threshold: threshold === "" ? undefined : Number(threshold),
        operator: operator || undefined,
        effective_date: effectiveDate || undefined,
        resolved_concerns: Object.entries(resolved)
          .filter(([, v]) => v)
          .map(([k]) => k),
      }),
    );
  }

  return (
    <div className="mt-3 space-y-4" data-testid="proposal-detail">
      <div className="rounded-xl border border-stone-200 bg-stone-50 px-4 py-3">
        <p className="text-xs uppercase tracking-wide text-stone-400">
          {copy.concentration.sourceWording}
        </p>
        <blockquote className="mt-1 text-sm text-stone-800">
          “{p.evidence.source_text}”
        </blockquote>
        <p className="mt-1 text-xs text-stone-500">
          {p.evidence.locator && `${p.evidence.locator} · `}
          {p.evidence.source_reference}
        </p>
      </div>

      <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
        <KV label={copy.concentration.proposedMetric} value={p.proposed_metric_id || "—"} />
        <KV label={copy.concentration.matchOutcome} value={p.match_outcome} />
        <KV
          label={copy.concentration.threshold}
          value={
            p.extracted_threshold === null
              ? "—"
              : `${p.extracted_operator === "min" ? "≥" : "≤"} ${p.extracted_threshold}${p.extracted_unit === "percent" ? "%" : ""}`
          }
        />
        <KV
          label={copy.concentration.confidence}
          value={`extraction ${(p.extraction_confidence * 100).toFixed(0)}% · mapping ${(p.mapping_confidence * 100).toFixed(0)}%`}
        />
        {p.denominator_basis && (
          <KV label="Denominator" value={p.denominator_basis} />
        )}
        {p.weighting_basis && <KV label="Weighting" value={p.weighting_basis} />}
      </dl>

      {p.concern_codes.length > 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-xs font-medium uppercase tracking-wide text-amber-700">
            {copy.concentration.concerns}
          </p>
          <ul className="mt-1 space-y-1 text-sm text-amber-800">
            {p.concern_codes.map((code) => (
              <li key={code} className="flex items-center justify-between gap-3">
                <span>{code.replace(/_/g, " ")}</span>
                {!decided && (
                  <label className="flex items-center gap-1 text-xs">
                    <input
                      type="checkbox"
                      checked={resolved[code] ?? false}
                      onChange={(e) =>
                        setResolved((r) => ({ ...r, [code]: e.target.checked }))
                      }
                    />
                    resolved
                  </label>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {p.confirmation_questions.length > 0 && (
        <div>
          <p className="text-xs uppercase tracking-wide text-stone-400">
            {copy.concentration.questions}
          </p>
          {p.confirmation_questions.map((q) => (
            <div key={q} className="mt-2 rounded-xl border border-stone-200 px-4 py-3">
              <p className="text-sm text-stone-800">{q}</p>
              <div className="mt-2 flex items-end gap-2">
                <label className="block grow">
                  <span className="text-xs uppercase tracking-wide text-stone-400">
                    {copy.concentration.answerLabel}
                  </span>
                  <input
                    value={answers[q] ?? ""}
                    onChange={(e) =>
                      setAnswers((a) => ({ ...a, [q]: e.target.value }))
                    }
                    className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
                  />
                </label>
                <SecondaryButton
                  disabled={busy || !(answers[q] ?? "").trim()}
                  onClick={() =>
                    void act(() =>
                      ops.answerConcentrationQuestion(
                        clientId, p.proposal_id, q, answers[q]),
                    )
                  }
                >
                  {copy.concentration.recordAnswer}
                </SecondaryButton>
              </div>
            </div>
          ))}
        </div>
      )}

      {p.confirmation_answers.length > 0 && (
        <div className="text-sm text-stone-600">
          {p.confirmation_answers.map((a) => (
            <p key={a.question}>
              <span className="text-stone-400">A:</span> {a.answer}{" "}
              <span className="text-xs text-stone-400">
                ({a.answered_by}, {formatDate(a.answered_at)})
              </span>
            </p>
          ))}
        </div>
      )}

      {!decided && (
        <div className="rounded-xl border border-stone-200 px-4 py-3">
          <p className="text-xs uppercase tracking-wide text-stone-400">
            {copy.concentration.editHeading}
          </p>
          <div className="mt-2 grid gap-3 sm:grid-cols-3">
            <label className="block">
              <span className="text-xs text-stone-500">
                {copy.concentration.threshold}
              </span>
              <input
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                inputMode="decimal"
                className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
              />
            </label>
            <label className="block">
              <span className="text-xs text-stone-500">
                {copy.concentration.operator}
              </span>
              <select
                value={operator}
                onChange={(e) => setOperator(e.target.value)}
                className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
              >
                <option value="">—</option>
                <option value="max">max (must not exceed)</option>
                <option value="min">min (must not fall below)</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs text-stone-500">
                {copy.concentration.effectiveDate}
              </span>
              <input
                type="date"
                value={effectiveDate}
                onChange={(e) => setEffectiveDate(e.target.value)}
                className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
              />
            </label>
          </div>
          <label className="mt-3 block">
            <span className="text-xs text-stone-500">
              {copy.concentration.parametersJson}
            </span>
            <textarea
              value={paramsJson}
              onChange={(e) => setParamsJson(e.target.value)}
              rows={4}
              className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 font-mono text-xs"
            />
          </label>
          <div className="mt-3">
            <SecondaryButton disabled={busy} onClick={saveEdits}>
              {copy.concentration.saveEdits}
            </SecondaryButton>
          </div>
        </div>
      )}

      {!decided && blockers.length > 0 && (
        <div
          className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3"
          data-testid="approval-blockers"
        >
          <p className="text-xs font-medium uppercase tracking-wide text-amber-700">
            {copy.concentration.approvalBlockedHeading}
          </p>
          <ul className="mt-1 list-inside list-disc text-sm text-amber-800">
            {blockers.map((b) => (
              <li key={b}>{b}</li>
            ))}
          </ul>
        </div>
      )}

      {!decided && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-end gap-2">
            <label className="block grow">
              <span className="text-xs text-stone-500">
                {copy.concentration.approveComments}
              </span>
              <input
                value={approveComments}
                onChange={(e) => setApproveComments(e.target.value)}
                className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
              />
            </label>
            <PrimaryButton
              disabled={busy || !isAdmin || blockers.length > 0}
              onClick={() =>
                void act(() =>
                  ops.approveConcentrationProposal(clientId, p.proposal_id, {
                    effective_date: effectiveDate || undefined,
                    comments: approveComments,
                  }),
                )
              }
            >
              {copy.concentration.approve}
            </PrimaryButton>
          </div>
          <div className="flex flex-wrap items-end gap-2">
            <label className="block grow">
              <span className="text-xs text-stone-500">
                {copy.concentration.reasonLabel}
              </span>
              <input
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
              />
            </label>
            <SecondaryButton
              disabled={busy || !reason.trim()}
              onClick={() =>
                void act(() =>
                  ops.rejectConcentrationProposal(clientId, p.proposal_id, reason),
                )
              }
            >
              {copy.concentration.reject}
            </SecondaryButton>
            <SecondaryButton
              disabled={busy || !reason.trim()}
              onClick={() =>
                void act(() =>
                  ops.markConcentrationUnsupported(clientId, p.proposal_id, reason),
                )
              }
            >
              {copy.concentration.unsupported}
            </SecondaryButton>
            <SecondaryButton
              disabled={busy || !reason.trim()}
              onClick={() =>
                void act(() =>
                  ops.markConcentrationNotApplicable(clientId, p.proposal_id, reason),
                )
              }
            >
              {copy.concentration.notApplicable}
            </SecondaryButton>
          </div>
          <div className="flex flex-wrap items-end gap-2">
            <label className="block grow">
              <span className="text-xs text-stone-500">
                {copy.concentration.clarifyQuestion}
              </span>
              <input
                value={clarifyQ}
                onChange={(e) => setClarifyQ(e.target.value)}
                className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
              />
            </label>
            <SecondaryButton
              disabled={busy || !clarifyQ.trim()}
              onClick={() =>
                void act(() =>
                  ops.requestConcentrationClarification(
                    clientId, p.proposal_id, clarifyQ),
                )
              }
            >
              {copy.concentration.clarify}
            </SecondaryButton>
          </div>
        </div>
      )}

      {p.status === "approved" && (
        <div className="flex flex-wrap items-end gap-2">
          <p className="grow text-sm text-stone-600">
            Approved by {p.approval.operator} on{" "}
            {formatDate(p.approval.decided_at ?? "")}
            {p.approval.comments && <> — “{p.approval.comments}”</>}
          </p>
          <label className="block">
            <span className="text-xs text-stone-500">
              {copy.concentration.reasonLabel}
            </span>
            <input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="mt-1 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm"
            />
          </label>
          <SecondaryButton
            disabled={busy || !isAdmin || !reason.trim()}
            onClick={() =>
              void act(() =>
                ops.supersedeConcentrationProposal(clientId, p.proposal_id, reason),
              )
            }
          >
            {copy.concentration.supersede}
          </SecondaryButton>
        </div>
      )}

      {p.operator_notes.length > 0 && (
        <div>
          <p className="text-xs uppercase tracking-wide text-stone-400">
            {copy.concentration.auditHeading}
          </p>
          <ul className="mt-1 space-y-1 text-xs text-stone-500">
            {p.operator_notes.map((n, i) => (
              <li key={i}>
                {formatDate(n.at)} · {n.actor}: {n.note}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-stone-100 py-1">
      <dt className="text-xs uppercase tracking-wide text-stone-400">{label}</dt>
      <dd className="text-right text-stone-800">{value}</dd>
    </div>
  );
}
