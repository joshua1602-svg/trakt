import type { EvidenceItem, EvidenceTable } from "@/api/types";
import { copy } from "@/lib/copy";

function isTable(data: unknown): data is EvidenceTable {
  return (
    typeof data === "object" &&
    data !== null &&
    Array.isArray((data as EvidenceTable).columns) &&
    Array.isArray((data as EvidenceTable).rows)
  );
}

function EvidenceBody({ item }: { item: EvidenceItem }) {
  const { kind, data } = item;

  if (kind === "list" && Array.isArray(data)) {
    return (
      <ul className="list-disc space-y-1 pl-5 text-sm text-stone-600">
        {data.map((entry, i) => (
          <li key={i}>{String(entry)}</li>
        ))}
      </ul>
    );
  }

  if (kind === "table" && isTable(data)) {
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr>
              {data.columns.map((col) => (
                <th key={col} className="border-b border-stone-200 pb-2 pr-4 font-medium text-stone-500">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j} className="border-b border-stone-100 py-2 pr-4 text-stone-700">
                    {String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (typeof data === "object" && data !== null) {
    return (
      <dl className="space-y-1 text-sm">
        {Object.entries(data as Record<string, unknown>).map(([key, value]) => (
          <div key={key} className="flex gap-2">
            <dt className="font-medium text-stone-500">{key}</dt>
            <dd className="text-stone-700">{String(value)}</dd>
          </div>
        ))}
      </dl>
    );
  }

  return <p className="text-sm text-stone-600">{String(data)}</p>;
}

/** Collapsed-by-default supporting detail for a stage. */
export function EvidenceView({ items }: { items: EvidenceItem[] }) {
  if (items.length === 0) return null;
  return (
    <details className="group rounded-xl border border-stone-200 bg-stone-50 px-4 py-3">
      <summary className="cursor-pointer select-none text-sm font-medium text-stone-600 hover:text-stone-900">
        {copy.common.showDetails}
      </summary>
      <div className="mt-4 space-y-5">
        {items.map((item, i) => (
          <div key={i}>
            <h4 className="mb-2 text-sm font-medium text-stone-700">{item.label}</h4>
            <EvidenceBody item={item} />
          </div>
        ))}
      </div>
    </details>
  );
}
