import { useState } from "react";
import type { VersionRow } from "@/api/adminTypes";
import { copy } from "@/lib/copy";
import { formatDate } from "@/lib/format";
import { DialogButtons, Field, Modal } from "./primitives";

/**
 * Explicit confirmation before rolling back. The version being replaced is
 * kept, never deleted, and the dialog says so.
 */
export function RollbackDialog({
  packageLabel,
  activeVersion,
  candidates,
  busy,
  onCancel,
  onConfirm,
}: {
  packageLabel: string;
  activeVersion: number;
  candidates: VersionRow[];
  busy?: boolean;
  onCancel: () => void;
  onConfirm: (version: number) => void;
}) {
  const options = candidates
    .filter((v) => v.version !== activeVersion && v.status !== "draft")
    .sort((a, b) => b.version - a.version);
  const [target, setTarget] = useState<number | undefined>(options[0]?.version);

  return (
    <Modal labelledBy="rollback-heading">
      <h2 id="rollback-heading" className="text-lg font-semibold text-stone-900">
        {copy.admin.rollback.heading}
      </h2>

      <dl className="mt-4 grid grid-cols-2 gap-4">
        <Field label={copy.admin.activate.package}>{packageLabel}</Field>
        <Field label={copy.admin.activeVersion}>v{activeVersion}</Field>
      </dl>

      <label htmlFor="rollback-target" className="mt-5 block text-sm text-stone-800">
        {copy.admin.rollback.choose}
      </label>
      <select
        id="rollback-target"
        value={target ?? ""}
        onChange={(event) => setTarget(Number(event.target.value))}
        className="mt-2 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
      >
        {options.map((option) => (
          <option key={option.version} value={option.version}>
            {copy.common.version} {option.version}
            {option.activated_at ? ` · ${formatDate(option.activated_at)}` : ""}
          </option>
        ))}
      </select>

      <ul className="mt-5 space-y-2 text-sm text-stone-700">
        <li>{copy.admin.rollback.help}</li>
        <li>{copy.admin.activate.pinnedNote}</li>
        <li>{copy.admin.activate.futureNote}</li>
      </ul>

      <DialogButtons
        onCancel={onCancel}
        onConfirm={() => target !== undefined && onConfirm(target)}
        confirmLabel={copy.admin.rollback.button}
        busy={busy}
        destructive
        disabled={target === undefined}
      />
    </Modal>
  );
}
