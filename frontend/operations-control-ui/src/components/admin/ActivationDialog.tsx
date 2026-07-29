import { OctagonAlert } from "lucide-react";
import type { ActivationBlocker, ImpactAnalysis } from "@/api/adminTypes";
import { copy } from "@/lib/copy";
import { DialogButtons, Field, Modal } from "./primitives";

/**
 * Explicit confirmation before activation. It always states which package,
 * which version, which version it replaces, that running workflows stay
 * pinned, and that only future runs use the new version.
 */
export function ActivationDialog({
  packageLabel,
  version,
  replacesVersion,
  impact,
  blockers,
  canActivate,
  invalidReason,
  busy,
  onCancel,
  onConfirm,
}: {
  packageLabel: string;
  version: number;
  replacesVersion: number;
  impact?: ImpactAnalysis;
  blockers: ActivationBlocker[];
  canActivate: boolean;
  invalidReason?: string;
  busy?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <Modal labelledBy="activate-heading">
      <h2 id="activate-heading" className="text-lg font-semibold text-stone-900">
        {copy.admin.activate.heading}
      </h2>

      <dl className="mt-4 grid grid-cols-3 gap-4">
        <Field label={copy.admin.activate.package}>{packageLabel}</Field>
        <Field label={copy.admin.activate.version}>v{version}</Field>
        <Field label={copy.admin.activate.replaces}>v{replacesVersion}</Field>
      </dl>

      <ul className="mt-5 space-y-2 text-sm text-stone-700">
        <li>{copy.admin.activate.pinnedNote}</li>
        <li>{copy.admin.activate.futureNote}</li>
        <li>{copy.admin.activate.historyNote}</li>
      </ul>

      {impact && impact.pinned_workflows > 0 && (
        <p className="mt-4 rounded-xl bg-stone-50 px-3 py-2 text-sm text-stone-700">
          {copy.admin.impact.running}: <strong>{impact.running_pinned_workflows}</strong> ·{" "}
          {copy.admin.impact.clients}: <strong>{impact.clients}</strong> ·{" "}
          {copy.admin.impact.portfolios}: <strong>{impact.portfolios}</strong>
        </p>
      )}

      {(blockers.length > 0 || !canActivate) && (
        <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-3 py-3 text-sm text-rose-900">
          <p className="flex items-start gap-2 font-medium">
            <OctagonAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {copy.admin.activate.blockedHeading}
          </p>
          <ul className="mt-2 space-y-1">
            {blockers.map((blocker) => (
              <li key={blocker.message}>{blocker.message}</li>
            ))}
            {blockers.length === 0 && !canActivate && (
              <li>{invalidReason || copy.admin.activate.invalidHelp}</li>
            )}
          </ul>
        </div>
      )}

      <DialogButtons
        onCancel={onCancel}
        onConfirm={onConfirm}
        confirmLabel={copy.admin.activate.button}
        busy={busy}
        disabled={!canActivate || blockers.length > 0}
      />
    </Modal>
  );
}
