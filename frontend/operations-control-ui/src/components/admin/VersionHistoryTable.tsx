import clsx from "clsx";
import type { VersionRow } from "@/api/adminTypes";
import { StatusChip } from "@/components/StatusChip";
import { copy } from "@/lib/copy";
import { formatDate } from "@/lib/format";
import { EmptyNote } from "./primitives";

/** Every version of a package, newest first, with the active one highlighted. */
export function VersionHistoryTable({
  versions,
  activeVersion,
  selected,
  onSelect,
}: {
  versions: VersionRow[];
  activeVersion: number;
  selected?: number;
  onSelect?: (version: number) => void;
}) {
  if (versions.length === 0) {
    return <EmptyNote>{copy.admin.files.empty}</EmptyNote>;
  }
  const rows = [...versions].sort((a, b) => b.version - a.version);
  return (
    <div className="overflow-x-auto rounded-2xl border border-stone-200 bg-white">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-stone-200 text-xs text-stone-500">
          <tr>
            <th scope="col" className="px-4 py-3 font-medium">
              {copy.common.version}
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              {copy.admin.status}
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              {copy.admin.createdBy}
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              {copy.admin.activatedAt}
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              {copy.admin.notes}
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const isActive = row.version === activeVersion;
            return (
              <tr
                key={row.version}
                data-version={row.version}
                data-active={isActive ? "true" : undefined}
                onClick={onSelect ? () => onSelect(row.version) : undefined}
                className={clsx(
                  "border-b border-stone-100 last:border-0",
                  isActive && "bg-emerald-50/60",
                  selected === row.version && "ring-2 ring-inset ring-stone-900/20",
                  onSelect && "cursor-pointer hover:bg-stone-50",
                )}
              >
                <th scope="row" className="px-4 py-3 font-medium text-stone-900">
                  v{row.version}
                  {isActive && (
                    <span className="ml-2 text-xs font-normal text-emerald-700">
                      {copy.admin.activeVersion.toLowerCase()}
                    </span>
                  )}
                </th>
                <td className="px-4 py-3">
                  <StatusChip status={row.status} />
                </td>
                <td className="px-4 py-3 text-stone-600">{row.created_by || "—"}</td>
                <td className="px-4 py-3 text-stone-600">
                  {formatDate(row.activated_at) || "—"}
                </td>
                <td className="px-4 py-3 text-stone-500">{row.notes || "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
