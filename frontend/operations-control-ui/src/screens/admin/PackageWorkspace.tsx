import { useState, type ReactNode } from "react";
import { OpsError } from "@/api/OpsClient";
import { useOpsClient } from "@/api/context";
import type { ActivationBlocker, ConfigLayer, LayerOverview } from "@/api/adminTypes";
import { AccessDenied } from "@/components/admin/AccessDenied";
import { ActivationDialog } from "@/components/admin/ActivationDialog";
import { AuditTimeline } from "@/components/admin/AuditTimeline";
import { ConfigurationDiff } from "@/components/admin/ConfigurationDiff";
import { DependencyPanel } from "@/components/admin/DependencyPanel";
import { FileInspector } from "@/components/admin/FileInspector";
import { ImpactPanel } from "@/components/admin/ImpactPanel";
import { RollbackDialog } from "@/components/admin/RollbackDialog";
import { ValidationResultPanel } from "@/components/admin/ValidationResultPanel";
import { VersionHistoryTable } from "@/components/admin/VersionHistoryTable";
import {
  EmptyNote,
  Field,
  Modal,
  DialogButtons,
  SectionHeading,
  TechnicalDetails,
} from "@/components/admin/primitives";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { StatusChip } from "@/components/StatusChip";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { formatDate } from "@/lib/format";
import { useAdminLoad } from "@/lib/useAdminLoad";
import { errorMessage, useLoad } from "@/lib/useLoad";

type Section = "summary" | "dependencies" | "files" | "history" | "compare" | "impact";

const SECTIONS: { key: Section; label: string }[] = [
  { key: "summary", label: copy.admin.sections.summary },
  { key: "dependencies", label: copy.admin.sections.dependencies },
  { key: "files", label: copy.admin.sections.files },
  { key: "compare", label: copy.admin.sections.compare },
  { key: "impact", label: copy.admin.sections.impact },
  { key: "history", label: copy.admin.sections.history },
];

function CompareSection({ layer, info }: { layer: ConfigLayer; info: LayerOverview }) {
  const client = useOpsClient();
  const versions = [...info.versions].sort((a, b) => b.version - a.version);
  const [from, setFrom] = useState(info.active_version);
  const [to, setTo] = useState(info.draft?.version ?? info.active_version);
  const { data, error, loading, reload } = useLoad(
    () => client.compareConfigVersions(layer, from, to),
    [layer, from, to],
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4">
        {(
          [
            { id: "compare-from", label: copy.admin.compare.from, value: from, set: setFrom },
            { id: "compare-to", label: copy.admin.compare.to, value: to, set: setTo },
          ] as const
        ).map((picker) => (
          <div key={picker.id}>
            <label htmlFor={picker.id} className="block text-xs text-stone-500">
              {picker.label}
            </label>
            <select
              id={picker.id}
              value={picker.value}
              onChange={(event) => picker.set(Number(event.target.value))}
              className="mt-1 rounded-xl border border-stone-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            >
              {versions.map((version) => (
                <option key={version.version} value={version.version}>
                  {copy.common.version} {version.version}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
      {loading && <Loading />}
      {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
      {data && !loading && <ConfigurationDiff comparison={data} />}
    </div>
  );
}

function ImpactSection({ layer, version }: { layer: ConfigLayer; version: number }) {
  const client = useOpsClient();
  const { data, error, loading, reload } = useLoad(
    () => client.getConfigImpact(layer, version),
    [layer, version],
  );
  if (loading) return <Loading />;
  if (error) return <ErrorNote message={error} onRetry={() => void reload()} />;
  return data ? <ImpactPanel impact={data} /> : null;
}

function HistorySection({ layer, info }: { layer: ConfigLayer; info: LayerOverview }) {
  const client = useOpsClient();
  const { data, error, loading, reload } = useLoad(() => client.getConfigAudit(), []);
  return (
    <div className="space-y-6">
      <div>
        <SectionHeading>{copy.admin.priorVersions}</SectionHeading>
        <VersionHistoryTable versions={info.versions} activeVersion={info.active_version} />
      </div>
      <div>
        <SectionHeading>{copy.admin.audit.title}</SectionHeading>
        {loading && <Loading />}
        {error && !loading && <ErrorNote message={error} onRetry={() => void reload()} />}
        {data && !loading && (
          <AuditTimeline entries={data.entries.filter((entry) => entry.layer === layer)} />
        )}
      </div>
    </div>
  );
}

/**
 * The lifecycle workspace for one configuration package layer: review the
 * active version, create and check a draft, compare, activate, roll back.
 *
 * Every state-changing action goes through a confirmation, shows a loading
 * state, waits for the backend, reports the outcome and refreshes the package
 * state. Nothing here mutates an active version.
 */
export function PackageWorkspace({
  layer,
  title,
  subtitle,
  intro,
}: {
  layer: ConfigLayer;
  title: string;
  subtitle: string;
  intro?: ReactNode;
}) {
  const client = useOpsClient();
  const toast = useToast();
  const { data, error, forbidden, loading, reload } = useAdminLoad(
    () => client.getConfigOverview(),
    [],
  );
  const [section, setSection] = useState<Section>("summary");
  const [confirmDraft, setConfirmDraft] = useState(false);
  const [activateFor, setActivateFor] = useState<number | null>(null);
  const [rollbackOpen, setRollbackOpen] = useState(false);
  const [blockers, setBlockers] = useState<ActivationBlocker[]>([]);
  const [busy, setBusy] = useState(false);

  if (forbidden) return <AccessDenied />;
  if (loading && !data) return <Loading />;
  if (!data) {
    return <ErrorNote message={copy.admin.unavailable} onRetry={() => void reload()} />;
  }

  const info = data.layers[layer];
  const draft = info.draft;

  async function run(action: () => Promise<unknown>, successMessage: string) {
    setBusy(true);
    try {
      await action();
      toast.show(successMessage, "success");
      await reload({ quiet: true });
    } catch (err) {
      if (err instanceof OpsError && err.blockers.length > 0) {
        setBlockers(err.blockers);
      }
      toast.show(errorMessage(err), "error");
    } finally {
      setBusy(false);
    }
  }

  const canActivateDraft = Boolean(draft?.validated);

  return (
    <>
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-stone-900">{title}</h1>
        <p className="mt-1 text-sm text-stone-500">{subtitle}</p>
      </header>
      {intro}

      {/* Active version */}
      <section className="rounded-2xl border border-stone-200 bg-white p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs text-stone-500">{copy.admin.activeVersion}</p>
            <p className="text-2xl font-semibold text-stone-900">v{info.active_version}</p>
          </div>
          <div className="flex items-center gap-2">
            <StatusChip status={info.status} />
            <StatusChip status={info.validated ? "valid" : "failed_checks"} />
          </div>
        </div>
        <dl className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Field label={copy.admin.activatedAt}>{formatDate(info.activated_at) || "—"}</Field>
          <Field label={copy.admin.activatedBy}>{info.activated_by || "—"}</Field>
          <Field label={copy.admin.partCount}>{info.file_count}</Field>
          <Field label={copy.admin.priorVersions}>{info.versions.length}</Field>
        </dl>
        <TechnicalDetails>
          <p>
            {copy.admin.fingerprint}: {info.package_hash}
          </p>
        </TechnicalDetails>
      </section>

      {/* Draft */}
      <section className="mt-6">
        <SectionHeading>{copy.admin.draft.heading}</SectionHeading>
        {draft ? (
          <div className="space-y-4 rounded-2xl border border-dashed border-amber-300 bg-amber-50/50 p-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-xl font-semibold text-stone-900">v{draft.version}</p>
                <p className="mt-0.5 text-sm text-stone-600">
                  {copy.admin.basedOn} {draft.based_on_version ?? "—"}
                </p>
              </div>
              <StatusChip status="draft" />
            </div>
            <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Field label={copy.admin.createdBy}>{draft.created_by || "—"}</Field>
              <Field label={copy.admin.createdAt}>{formatDate(draft.created_at) || "—"}</Field>
              <Field label={copy.admin.partCount}>{draft.file_count}</Field>
              <Field label={copy.admin.notes}>{draft.notes || copy.admin.noNotes}</Field>
            </dl>

            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                {copy.admin.draft.changedParts}
              </h3>
              {draft.changed_files.length === 0 ? (
                <p className="mt-1 text-sm text-stone-500">{copy.admin.draft.noChanges}</p>
              ) : (
                <ul className="mt-1 flex flex-wrap gap-2">
                  {draft.changed_files.map((file) => (
                    <li
                      key={file.path}
                      className="rounded-full border border-stone-300 bg-white px-2.5 py-0.5 text-xs text-stone-700"
                    >
                      {file.label}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <ValidationResultPanel summary={draft.validation_summary} />

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                disabled={busy}
                onClick={() =>
                  void run(
                    () => client.validateConfigVersion(layer, draft.version),
                    copy.admin.validation.done,
                  )
                }
                className="rounded-xl bg-stone-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-stone-700 disabled:opacity-40"
              >
                {copy.admin.actions.validate}
              </button>
              <button
                type="button"
                disabled={busy || !canActivateDraft}
                title={canActivateDraft ? undefined : copy.admin.activate.invalidHelp}
                onClick={() => setActivateFor(draft.version)}
                className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-40"
              >
                {copy.admin.actions.activate}
              </button>
              <button
                type="button"
                onClick={() => setSection("compare")}
                className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 hover:bg-stone-50"
              >
                {copy.admin.actions.compare}
              </button>
            </div>
          </div>
        ) : (
          <EmptyNote>{copy.admin.draft.none}</EmptyNote>
        )}

        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            disabled={busy}
            onClick={() => setConfirmDraft(true)}
            className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-40"
          >
            {copy.admin.actions.createDraft}
          </button>
          <button
            type="button"
            disabled={busy || info.versions.filter((v) => v.status !== "draft").length < 2}
            onClick={() => setRollbackOpen(true)}
            className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-40"
          >
            {copy.admin.actions.rollback}
          </button>
        </div>
        <p className="mt-3 text-xs text-stone-500">{copy.admin.draft.readOnlyNote}</p>
      </section>

      {/* Sections */}
      <nav
        aria-label={copy.admin.sections.summary}
        className="mt-10 flex gap-1 overflow-x-auto border-b border-stone-200 pb-px"
      >
        {SECTIONS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => setSection(tab.key)}
            aria-current={section === tab.key ? "page" : undefined}
            className={
              section === tab.key
                ? "shrink-0 border-b-2 border-stone-900 px-4 py-2 text-sm font-medium text-stone-900"
                : "shrink-0 border-b-2 border-transparent px-4 py-2 text-sm font-medium text-stone-500 hover:text-stone-800"
            }
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="mt-6">
        {section === "summary" && (
          <div className="space-y-6">
            <ValidationResultPanel summary={info.validation_summary} />
            <div>
              <SectionHeading>{copy.admin.files.heading}</SectionHeading>
              <ul className="flex flex-wrap gap-2">
                {info.files.map((file) => (
                  <li
                    key={file.path}
                    className="rounded-full border border-stone-200 bg-white px-3 py-1 text-xs text-stone-700"
                  >
                    {file.label}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
        {section === "dependencies" && <DependencyPanel dependencies={info.dependencies} />}
        {section === "files" && (
          <FileInspector layer={layer} version={info.active_version} files={info.files} />
        )}
        {section === "compare" && <CompareSection layer={layer} info={info} />}
        {section === "impact" && <ImpactSection layer={layer} version={info.active_version} />}
        {section === "history" && <HistorySection layer={layer} info={info} />}
      </div>

      {error && data && <div className="mt-6"><ErrorNote message={error} onRetry={() => void reload()} /></div>}

      {confirmDraft && (
        <Modal labelledBy="draft-heading">
          <h2 id="draft-heading" className="text-lg font-semibold text-stone-900">
            {copy.admin.draft.createPrompt}
          </h2>
          <p className="mt-2 text-sm text-stone-600">{copy.admin.draft.createHelp}</p>
          <DialogButtons
            onCancel={() => setConfirmDraft(false)}
            onConfirm={() => {
              setConfirmDraft(false);
              void run(() => client.createConfigDraft(layer), copy.admin.draft.created);
            }}
            confirmLabel={copy.admin.draft.createButton}
            busy={busy}
          />
        </Modal>
      )}

      {activateFor !== null && (
        <ActivationDialog
          packageLabel={info.label}
          version={activateFor}
          replacesVersion={info.active_version}
          blockers={blockers}
          canActivate={canActivateDraft}
          busy={busy}
          onCancel={() => {
            setActivateFor(null);
            setBlockers([]);
          }}
          onConfirm={() => {
            const version = activateFor;
            setActivateFor(null);
            void run(
              () => client.activateConfigVersion(layer, version),
              copy.admin.activate.done,
            );
          }}
        />
      )}

      {rollbackOpen && (
        <RollbackDialog
          packageLabel={info.label}
          activeVersion={info.active_version}
          candidates={info.versions}
          busy={busy}
          onCancel={() => setRollbackOpen(false)}
          onConfirm={(version) => {
            setRollbackOpen(false);
            void run(() => client.rollbackConfig(layer, version), copy.admin.rollback.done);
          }}
        />
      )}
    </>
  );
}
