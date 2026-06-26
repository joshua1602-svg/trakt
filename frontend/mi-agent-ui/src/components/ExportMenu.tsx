import { useEffect, useRef, useState, type RefObject } from "react";
import { Check, Download, FileImage, FileJson, FileSpreadsheet, FileText, Shapes } from "lucide-react";
import type { ChartArtifact, TableArtifact } from "@/domain";
import { isChartArtifact } from "@/domain";
import { IconButton } from "@/components/ui";
import { formatHeading, formatUiTitle } from "@/lib/utils";
import {
  type CellValue,
  chartPngBlob,
  chartSvgBlob,
  downloadBlob,
  exportFilename,
  findChartSvg,
  toCsv,
  toXlsxBlob,
} from "@/lib/export";

type ExportArtifact = ChartArtifact | TableArtifact;

/** Flatten an artifact's rows into a header + cell-matrix for CSV/XLSX. */
function artifactToTable(artifact: ExportArtifact): { headers: string[]; rows: CellValue[][] } {
  if (isChartArtifact(artifact)) {
    const keys = Array.from(
      new Set(
        [
          artifact.xKey,
          ...artifact.series.map((s) => s.key),
          artifact.valueKey,
          artifact.yKey,
          artifact.sizeKey,
        ].filter((k): k is string => !!k),
      ),
    );
    const headers = keys.map((k) => {
      const s = artifact.series.find((ser) => ser.key === k);
      if (s) return formatHeading(s.label);
      if (k === artifact.yKey && artifact.yLabel) return formatHeading(artifact.yLabel);
      if (k === artifact.xKey && artifact.xLabel) return formatHeading(artifact.xLabel);
      return formatUiTitle(k);
    });
    const rows = artifact.rows.map((r) => keys.map((k) => r[k] ?? ""));
    return { headers, rows };
  }
  const headers = artifact.columns.map((c) => formatHeading(c.label));
  const rows = artifact.rows.map((r) => artifact.columns.map((c) => r[c.key] ?? ""));
  return { headers, rows };
}

interface ExportItem {
  key: string;
  label: string;
  icon: typeof Download;
  run: () => void | Promise<void>;
}

/**
 * Compact export control for an MI Agent chart/table result. A single Download
 * icon opens a small menu of presentation-friendly formats — image exports for
 * charts (PNG/SVG), data exports for both (CSV/XLSX) — so the default result
 * view stays uncluttered. Every file gets a clean, timestamped name.
 */
export function ExportMenu({
  artifact,
  bodyRef,
  onJson,
}: {
  artifact: ExportArtifact;
  bodyRef: RefObject<HTMLElement | null>;
  onJson?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const title = formatHeading(artifact.title);

  const flash = (key: string) => {
    setDone(key);
    setTimeout(() => setDone((d) => (d === key ? null : d)), 1400);
  };

  const items: ExportItem[] = [];

  if (isChartArtifact(artifact)) {
    const svg = () => findChartSvg(bodyRef.current);
    if (svg()) {
      items.push({
        key: "png",
        label: "PNG image",
        icon: FileImage,
        run: async () => {
          const el = svg();
          if (!el) return;
          downloadBlob(await chartPngBlob(el), exportFilename(title, "png"));
          flash("png");
        },
      });
      items.push({
        key: "svg",
        label: "SVG vector",
        icon: Shapes,
        run: () => {
          const el = svg();
          if (!el) return;
          downloadBlob(chartSvgBlob(el), exportFilename(title, "svg"));
          flash("svg");
        },
      });
    }
  }

  items.push({
    key: "csv",
    label: "CSV data",
    icon: FileText,
    run: () => {
      const { headers, rows } = artifactToTable(artifact);
      downloadBlob(new Blob([toCsv(headers, rows)], { type: "text/csv;charset=utf-8" }), exportFilename(title, "csv"));
      flash("csv");
    },
  });

  if (!isChartArtifact(artifact)) {
    items.push({
      key: "xlsx",
      label: "Excel (XLSX)",
      icon: FileSpreadsheet,
      run: () => {
        const { headers, rows } = artifactToTable(artifact);
        downloadBlob(toXlsxBlob(headers, rows, title), exportFilename(title, "xlsx"));
        flash("xlsx");
      },
    });
  }

  if (onJson) {
    items.push({ key: "json", label: "JSON spec", icon: FileJson, run: onJson });
  }

  return (
    <div ref={containerRef} className="relative">
      <IconButton label="Export" active={open} onClick={() => setOpen((o) => !o)}>
        <Download size={14} />
      </IconButton>
      {open && (
        <div className="absolute right-0 z-20 mt-1 w-44 overflow-hidden rounded-lg border border-[var(--color-line)] bg-navy-900 py-1 shadow-xl">
          {items.map((it) => (
            <button
              key={it.key}
              type="button"
              onClick={() => {
                void it.run();
              }}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[12px] text-ink-200 transition-colors hover:bg-navy-800"
            >
              {done === it.key ? <Check size={13} className="text-mint-400" /> : <it.icon size={13} className="text-peri-300" />}
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
