"use client";

import { useState } from "react";

import { columnFormat, columnLabel, formatValue } from "@/lib/format";
import { cx } from "@/components/ui";
import type {
  ChartArtifact,
  DemoArtifact,
  KpiArtifact,
  TableArtifact,
} from "@/types/demo";

/**
 * Renders the artifacts the Trakt engine produced for one answer.
 *
 * Presentation only: every value arrives pre-computed, and the format contract
 * comes from the engine's own column metadata.
 *
 * The bar chart is hand-built rather than drawn by a charting library. A
 * horizontal bar list of a metric by dimension is the only chart shape these
 * answers need, and building it directly keeps the marketing bundle free of a
 * ~100 kB charting dependency, reads correctly from 320 px upward (a library's
 * fixed category axis does not), reserves its own height so it cannot shift
 * layout, and stays legible to a screen reader as a description list. The
 * product workspace (`frontend/mi-agent-ui`) keeps Recharts/Plotly for the
 * interactive charts that genuinely need them.
 */

function KpiBlock({ artifact }: { artifact: KpiArtifact }) {
  return (
    <div>
      <dl className="grid gap-3 sm:grid-cols-2">
        {artifact.kpis.map((kpi) => (
          <div key={kpi.label} className="rounded-lg border border-line bg-navy-950/50 p-4">
            <dt className="text-[11px] uppercase tracking-wider text-ink-400">
              {kpi.label}
            </dt>
            <dd className="mt-1.5 text-2xl font-semibold tabular-nums text-ink-100">
              {kpi.value}
            </dd>
          </div>
        ))}
      </dl>
      {artifact.coverage != null ? <Coverage value={artifact.coverage} /> : null}
    </div>
  );
}

function Coverage({ value }: { value: number }) {
  return (
    <p className="mt-2.5 text-[11px] text-ink-500">
      Balance coverage {formatValue(value, "pct")} of the funded book.
    </p>
  );
}

function ChartBlock({ artifact }: { artifact: ChartArtifact }) {
  const valueKey = artifact.valueKey ?? artifact.columns[1]?.key;
  if (!valueKey) {
    return <TableBlock artifact={{ ...artifact, kind: "table" }} />;
  }

  const format = columnFormat(artifact.columns, valueKey);
  const values = artifact.rows.map((row) => {
    const raw = row[valueKey];
    return typeof raw === "number" ? raw : 0;
  });
  const max = Math.max(...values, 1);

  return (
    <div>
      {artifact.title ? (
        <h4 className="mb-3 text-sm font-medium text-ink-200">{artifact.title}</h4>
      ) : null}

      <dl className="space-y-2.5">
        {artifact.rows.map((row, index) => {
          const value = values[index] ?? 0;
          const width = `${Math.max(1.5, (value / max) * 100)}%`;
          return (
            <div key={index}>
              <div className="flex items-baseline justify-between gap-3">
                <dt className="min-w-0 truncate text-xs text-ink-300">
                  {String(row[artifact.xKey] ?? "—")}
                </dt>
                <dd className="shrink-0 text-xs font-medium tabular-nums text-ink-100">
                  {formatValue(value, format, { compact: true })}
                </dd>
              </div>
              <div
                aria-hidden="true"
                className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-navy-800"
              >
                <div
                  className={cx(
                    "h-full rounded-full",
                    index === 0 ? "bg-peri-200" : "bg-peri-400",
                  )}
                  style={{ width }}
                />
              </div>
            </div>
          );
        })}
      </dl>

      <TableBlock
        artifact={{ ...artifact, kind: "table", title: null }}
        caption={`${artifact.title ?? "Result"} — full values`}
        collapsible
      />
    </div>
  );
}

/**
 * Row detail that used to sit in the answer prose.
 *
 * A real toggle, not a `title` attribute: native tooltips never open on
 * touch, which would have made this unreachable on a phone — and it is the
 * explanation that makes the two-totals answer land.
 */
function RowNote({ value, note }: { value: string; note: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
        className="text-left underline decoration-ink-500 decoration-dotted underline-offset-4 hover:decoration-ink-300"
      >
        {value}
      </button>
      {open ? (
        <span className="mt-1.5 block max-w-xs text-[11px] leading-relaxed text-ink-400">
          {note}
        </span>
      ) : null}
    </>
  );
}

function TableBlock({
  artifact,
  caption,
  collapsible = false,
}: {
  artifact: TableArtifact;
  caption?: string;
  collapsible?: boolean;
}) {
  const table = (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] border-collapse text-sm">
        {caption ? <caption className="sr-only">{caption}</caption> : null}
        <thead>
          <tr className="border-b border-line">
            {artifact.columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={cx(
                  "px-3 py-2 text-[11px] font-medium uppercase tracking-wider text-ink-400",
                  column.align === "right" ? "text-right" : "text-left",
                )}
              >
                {columnLabel(artifact.columns, column.key)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {artifact.rows.map((row, index) => (
            <tr key={index} className="border-b border-line-soft last:border-0">
              {artifact.columns.map((column, columnIndex) => {
                const value = formatValue(row[column.key] ?? null, column.format);
                const tooltip =
                  columnIndex === 0 && typeof row.tooltip === "string"
                    ? row.tooltip
                    : null;
                return (
                  <td
                    key={column.key}
                    className={cx(
                      "px-3 py-2 text-ink-200",
                      column.align === "right" ? "text-right tabular-nums" : "text-left",
                    )}
                  >
                    {tooltip ? <RowNote value={value} note={tooltip} /> : value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  if (!collapsible) {
    return (
      <div>
        {artifact.title ? (
          <h4 className="mb-2 text-sm font-medium text-ink-200">{artifact.title}</h4>
        ) : null}
        {table}
        {artifact.coverage != null ? <Coverage value={artifact.coverage} /> : null}
      </div>
    );
  }

  return (
    <details className="mt-3.5 rounded-lg border border-line-soft bg-navy-950/40">
      {/* A product capability — drill-through from a figure to its source —
          not a UI affordance. Closed by default so the panel stays compact. */}
      <summary className="cursor-pointer px-3 py-2 text-xs text-ink-400 hover:text-ink-200">
        Underlying values
      </summary>
      <div className="px-1 pb-2">{table}</div>
      {artifact.coverage != null ? (
        <p className="px-3 pb-2.5 text-[11px] text-ink-500">
          Balance coverage {formatValue(artifact.coverage, "pct")}.
        </p>
      ) : null}
    </details>
  );
}

export function ArtifactBlock({ artifact }: { artifact: DemoArtifact }) {
  switch (artifact.kind) {
    case "kpi":
      return <KpiBlock artifact={artifact} />;
    case "chart":
      return <ChartBlock artifact={artifact} />;
    case "table":
      return <TableBlock artifact={artifact} />;
    default:
      return null;
  }
}

export function ArtifactList({ artifacts }: { artifacts: DemoArtifact[] }) {
  if (artifacts.length === 0) return null;
  return (
    <div className="mt-4 space-y-5">
      {artifacts.map((artifact, index) => (
        <ArtifactBlock key={index} artifact={artifact} />
      ))}
    </div>
  );
}
