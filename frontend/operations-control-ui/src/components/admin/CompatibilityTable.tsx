import { AlertTriangle, Check, Minus } from "lucide-react";
import type { CompatibilityIssue, CompatibilityMatrix } from "@/api/adminTypes";
import { copy } from "@/lib/copy";
import { EmptyNote, SectionHeading } from "./primitives";

/**
 * Asset-to-regime compatibility. Only relationships the backend declares are
 * shown — nothing is inferred from a file name, and nothing is called
 * "planned" unless the backend says so.
 */
export function CompatibilityTable({
  matrix,
  issues,
}: {
  matrix: CompatibilityMatrix;
  issues?: CompatibilityIssue[];
}) {
  return (
    <section>
      <SectionHeading>{copy.admin.compatibility.heading}</SectionHeading>
      {matrix.rows.length === 0 ? (
        <EmptyNote>{copy.admin.assets.empty}</EmptyNote>
      ) : (
        <div className="overflow-x-auto rounded-2xl border border-stone-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-stone-200 text-xs text-stone-500">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium">
                  {copy.admin.compatibility.asset}
                </th>
                {matrix.regimes.map((regime) => (
                  <th key={regime.id} scope="col" className="px-4 py-3 font-medium">
                    {regime.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.rows.map((row) => (
                <tr key={row.asset} className="border-b border-stone-100 last:border-0">
                  <th scope="row" className="px-4 py-3 font-medium text-stone-900">
                    {row.asset_label}
                  </th>
                  {row.cells.map((cell) => (
                    <td
                      key={cell.regime}
                      data-supported={cell.supported ? "true" : "false"}
                      className="px-4 py-3"
                    >
                      <span
                        className={
                          cell.supported
                            ? "inline-flex items-center gap-1.5 text-emerald-700"
                            : "inline-flex items-center gap-1.5 text-stone-400"
                        }
                      >
                        {cell.supported ? (
                          <Check className="h-4 w-4" aria-hidden />
                        ) : (
                          <Minus className="h-4 w-4" aria-hidden />
                        )}
                        {cell.label}
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {issues && issues.length > 0 && (
        <div className="mt-4 space-y-2">
          {issues.map((issue) => (
            <p
              key={issue.message}
              className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              {issue.message}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
