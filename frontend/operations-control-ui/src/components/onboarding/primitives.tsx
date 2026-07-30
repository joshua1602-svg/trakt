import type { ReactNode } from "react";
import { AlertTriangle, Check, Info } from "lucide-react";
import clsx from "clsx";

/** A titled block of the wizard or the client editor. */
export function Card({
  title,
  description,
  actions,
  children,
}: {
  title?: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-6">
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-lg font-semibold text-stone-900">{title}</h2>}
            {description && <p className="mt-1 text-sm text-stone-500">{description}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function SectionHeading({ children }: { children: ReactNode }) {
  return (
    <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-stone-400">
      {children}
    </h3>
  );
}

/** A short explanatory note — never an error. */
export function Note({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  const Icon = tone === "warn" ? AlertTriangle : Info;
  return (
    <p
      className={clsx(
        "flex items-start gap-2 rounded-xl border px-4 py-3 text-sm leading-relaxed",
        tone === "warn"
          ? "border-amber-200 bg-amber-50 text-amber-900"
          : "border-stone-200 bg-stone-50 text-stone-600",
      )}
    >
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <span>{children}</span>
    </p>
  );
}

export function Field({
  label,
  hint,
  problem,
  origin,
  required,
  children,
}: {
  label: string;
  hint?: string;
  problem?: string;
  /** Where an adopted value came from. Shown so nothing appears from nowhere. */
  origin?: string;
  required?: boolean;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 flex items-baseline gap-2 text-sm font-medium text-stone-700">
        {label}
        {required && <span className="text-xs font-normal text-stone-400">Required</span>}
      </span>
      {children}
      {hint && !problem && <span className="mt-1 block text-xs text-stone-500">{hint}</span>}
      {origin && !problem && (
        <span className="mt-1 block text-xs text-stone-400">Taken from {origin}</span>
      )}
      {problem && <span className="mt-1 block text-xs text-red-600">{problem}</span>}
    </label>
  );
}

const INPUT =
  "w-full rounded-xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-900 " +
  "outline-none transition-colors focus:border-stone-900 disabled:bg-stone-50 disabled:text-stone-500";

export function TextInput({
  value,
  onChange,
  placeholder,
  disabled,
  invalid,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  invalid?: boolean;
}) {
  return (
    <input
      className={clsx(INPUT, invalid && "border-red-400")}
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

export function Select({
  value,
  onChange,
  options,
  placeholder,
  invalid,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
  invalid?: boolean;
}) {
  return (
    <select
      className={clsx(INPUT, invalid && "border-red-400")}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {placeholder !== undefined && <option value="">{placeholder}</option>}
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
}) {
  return (
    <label
      className={clsx(
        "flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3 transition-colors",
        disabled
          ? "cursor-not-allowed border-stone-200 bg-stone-50"
          : checked
            ? "border-stone-900 bg-stone-50"
            : "border-stone-200 hover:border-stone-400",
      )}
    >
      <input
        type="checkbox"
        className="mt-1 h-4 w-4 accent-stone-900"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>
        <span className="block text-sm font-medium text-stone-900">{label}</span>
        {description && <span className="mt-0.5 block text-xs text-stone-500">{description}</span>}
      </span>
    </label>
  );
}

export function PrimaryButton({
  children,
  onClick,
  disabled,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className="rounded-xl bg-stone-900 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-stone-700 disabled:cursor-not-allowed disabled:bg-stone-300"
    >
      {children}
    </button>
  );
}

export function SecondaryButton({
  children,
  onClick,
  disabled,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="rounded-xl border border-stone-300 px-4 py-2.5 text-sm font-medium text-stone-700 transition-colors hover:border-stone-500 disabled:cursor-not-allowed disabled:text-stone-400"
    >
      {children}
    </button>
  );
}

/**
 * create / update / unchanged, in plain words.
 *
 * `tense` matters: on the review screen nothing has happened yet, but in a
 * client's history it already has. Saying "will be created" about a version
 * that was written last March would be plainly wrong.
 */
export function ActionChip({ action, tense = "future" }: { action: string; tense?: "future" | "past" }) {
  const tone =
    action === "create"
      ? "bg-emerald-50 text-emerald-800 border-emerald-200"
      : action === "update"
        ? "bg-blue-50 text-blue-800 border-blue-200"
        : "bg-stone-100 text-stone-600 border-stone-200";
  const label =
    tense === "past"
      ? action === "create"
        ? "Created"
        : action === "update"
          ? "Changed"
          : "No change"
      : action === "create"
        ? "Will be created"
        : action === "update"
          ? "Will change"
          : "No change";
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        tone,
      )}
    >
      {action === "unchanged" && <Check className="h-3 w-3" aria-hidden />}
      {label}
    </span>
  );
}

export function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-stone-100 py-2 last:border-0">
      <span className="text-sm text-stone-500">{label}</span>
      <span className="text-sm font-medium text-stone-900">{value || "—"}</span>
    </div>
  );
}
