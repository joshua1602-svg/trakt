import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import clsx from "clsx";
import { copy } from "@/lib/copy";

/** Centred dialog, matching the workflow publish/hold dialogs. */
export function Modal({ children, labelledBy }: { children: ReactNode; labelledBy?: string }) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-stone-900/30 px-6">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        className="max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-2xl border border-stone-200 bg-white p-6 shadow-xl"
      >
        {children}
      </div>
    </div>
  );
}

export function DialogButtons({
  onCancel,
  onConfirm,
  confirmLabel,
  busy,
  destructive,
  disabled,
}: {
  onCancel: () => void;
  onConfirm: () => void;
  confirmLabel: string;
  busy?: boolean;
  destructive?: boolean;
  disabled?: boolean;
}) {
  return (
    <div className="mt-6 flex justify-end gap-3">
      <button
        type="button"
        onClick={onCancel}
        className="rounded-xl border border-stone-300 bg-white px-4 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50"
      >
        {copy.common.cancel}
      </button>
      <button
        type="button"
        disabled={busy || disabled}
        onClick={onConfirm}
        className={clsx(
          "rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-40",
          destructive ? "bg-stone-900 hover:bg-stone-700" : "bg-blue-600 hover:bg-blue-700",
        )}
      >
        {busy ? copy.common.loading : confirmLabel}
      </button>
    </div>
  );
}

/** Collapsible technical detail — never shown by default. */
export function TechnicalDetails({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-xs font-medium text-stone-500 hover:text-stone-700"
      >
        <ChevronDown className={clsx("h-3.5 w-3.5 transition-transform", open && "rotate-180")} aria-hidden />
        {open ? copy.admin.actions.hideTechnical : copy.admin.actions.showTechnical}
      </button>
      {open && (
        <div className="mt-2 overflow-x-auto rounded-xl bg-stone-50 px-3 py-2 font-mono text-xs text-stone-600">
          {children}
        </div>
      )}
    </div>
  );
}

/** Label/value pair used across the package screens. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-stone-500">{label}</dt>
      <dd className="mt-0.5 text-sm text-stone-900">{children}</dd>
    </div>
  );
}

export function EmptyNote({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-xl border border-dashed border-stone-200 px-4 py-8 text-center text-sm text-stone-400">
      {children}
    </p>
  );
}

export function SectionHeading({ children }: { children: ReactNode }) {
  return <h2 className="mb-3 text-sm font-semibold text-stone-700">{children}</h2>;
}
