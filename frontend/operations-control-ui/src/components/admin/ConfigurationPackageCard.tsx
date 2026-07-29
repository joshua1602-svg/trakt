import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import type { LayerOverview } from "@/api/adminTypes";
import { StatusChip } from "@/components/StatusChip";
import { copy } from "@/lib/copy";
import { formatDate } from "@/lib/format";
import { Field } from "./primitives";

/**
 * The at-a-glance state of one configuration package: what is active, whether
 * a draft is waiting, and whether the active version is valid.
 */
export function ConfigurationPackageCard({
  layer,
  to,
  extra,
}: {
  layer: LayerOverview;
  to?: string;
  extra?: ReactNode;
}) {
  const body = (
    <>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-base font-semibold text-stone-900">{layer.label}</p>
          <p className="mt-0.5 text-sm text-stone-500">{layer.description}</p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <StatusChip status={layer.status} />
          <StatusChip status={layer.validated ? "valid" : "failed_checks"} />
          {to && <ChevronRight className="h-4 w-4 text-stone-300" aria-hidden />}
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Field label={copy.admin.activeVersion}>
          <span className="font-semibold">v{layer.active_version}</span>
        </Field>
        <Field label={copy.admin.activatedAt}>{formatDate(layer.activated_at) || "—"}</Field>
        <Field label={copy.admin.activatedBy}>{layer.activated_by || "—"}</Field>
        <Field label={copy.admin.partCount}>{layer.file_count}</Field>
      </dl>

      {layer.draft ? (
        <p className="mt-4 flex flex-wrap items-center gap-2 rounded-xl border border-dashed border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          <StatusChip status="draft" />
          <span>
            {copy.admin.draftVersion} v{layer.draft.version} · {layer.draft.validation_summary.headline}
          </span>
        </p>
      ) : (
        <p className="mt-4 text-sm text-stone-400">{copy.admin.draft.none}</p>
      )}
      {extra}
    </>
  );

  const className =
    "block rounded-2xl border border-stone-200 bg-white p-6 text-left transition-colors";

  return to ? (
    <Link to={to} className={`${className} hover:border-stone-300 hover:bg-stone-50`}>
      {body}
    </Link>
  ) : (
    <div className={className}>{body}</div>
  );
}
