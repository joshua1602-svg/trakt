/**
 * Presentation-friendly exports for MI Agent outputs.
 *
 *  - Charts → PNG (PowerPoint-ready) and SVG, rasterised from the rendered
 *    chart's own <svg> so the export matches what the user sees.
 *  - Tables → CSV and a real (STORED-zip) XLSX, built from the artifact rows.
 *
 * Filenames are clean + timestamped, e.g. average_ltv_by_region_2026-06-26.png.
 * No third-party export dependency is pulled in.
 */

import { formatHeading, toFilenameStem } from "@/lib/utils";

/** Build a clean, dated filename: `<title-stem>_<yyyy-mm-dd>.<ext>`. */
export function exportFilename(title: string, ext: string, date = new Date()): string {
  const stem = toFilenameStem(formatHeading(title));
  return `${stem}_${date.toISOString().slice(0, 10)}.${ext}`;
}

/** Trigger a browser download for a Blob. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/* ------------------------------- CSV / XLSX ------------------------------ */

export type CellValue = string | number | null | undefined;

function csvCell(v: CellValue): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** Serialise a header row + data rows to CSV text (RFC 4180 quoting). */
export function toCsv(headers: string[], rows: CellValue[][]): string {
  const lines = [headers.map(csvCell).join(",")];
  for (const row of rows) lines.push(row.map(csvCell).join(","));
  return lines.join("\r\n");
}

function xmlEscape(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function sheetCell(ref: string, v: CellValue): string {
  if (v === null || v === undefined || v === "") return `<c r="${ref}"/>`;
  if (typeof v === "number" && Number.isFinite(v)) return `<c r="${ref}"><v>${v}</v></c>`;
  return `<c r="${ref}" t="inlineStr"><is><t xml:space="preserve">${xmlEscape(String(v))}</t></is></c>`;
}

function colRef(index: number): string {
  let n = index;
  let ref = "";
  do {
    ref = String.fromCharCode(65 + (n % 26)) + ref;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return ref;
}

function worksheetXml(headers: string[], rows: CellValue[][]): string {
  const all = [headers, ...rows];
  const xmlRows = all
    .map((row, r) => {
      const cells = row.map((v, c) => sheetCell(`${colRef(c)}${r + 1}`, v)).join("");
      return `<row r="${r + 1}">${cells}</row>`;
    })
    .join("");
  return (
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">` +
    `<sheetData>${xmlRows}</sheetData></worksheet>`
  );
}

/** Build the raw bytes of a minimal but valid .xlsx workbook (single sheet). */
export function toXlsxBytes(headers: string[], rows: CellValue[][], sheetName = "Sheet1"): Uint8Array {
  const safeName = xmlEscape(sheetName).slice(0, 31) || "Sheet1";
  const files: Array<[string, string]> = [
    [
      "[Content_Types].xml",
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
        `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">` +
        `<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>` +
        `<Default Extension="xml" ContentType="application/xml"/>` +
        `<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>` +
        `<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>` +
        `</Types>`,
    ],
    [
      "_rels/.rels",
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
        `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
        `<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>` +
        `</Relationships>`,
    ],
    [
      "xl/workbook.xml",
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
        `<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ` +
        `xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">` +
        `<sheets><sheet name="${safeName}" sheetId="1" r:id="rId1"/></sheets></workbook>`,
    ],
    [
      "xl/_rels/workbook.xml.rels",
      `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
        `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
        `<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>` +
        `</Relationships>`,
    ],
    ["xl/worksheets/sheet1.xml", worksheetXml(headers, rows)],
  ];
  const enc = new TextEncoder();
  return zipStore(files.map(([name, data]) => ({ name, data: enc.encode(data) })));
}

/** Build a minimal but valid .xlsx workbook Blob (single sheet, inline strings). */
export function toXlsxBlob(headers: string[], rows: CellValue[][], sheetName = "Sheet1"): Blob {
  const bytes = toXlsxBytes(headers, rows, sheetName);
  return new Blob([bytes as unknown as BlobPart], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
}

/* --------------------------- STORED-zip writer --------------------------- */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/** Pack files into a ZIP archive using STORED (no compression). */
export function zipStore(files: Array<{ name: string; data: Uint8Array }>): Uint8Array {
  const enc = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  const u16 = (n: number) => new Uint8Array([n & 0xff, (n >>> 8) & 0xff]);
  const u32 = (n: number) =>
    new Uint8Array([n & 0xff, (n >>> 8) & 0xff, (n >>> 16) & 0xff, (n >>> 24) & 0xff]);

  for (const file of files) {
    const nameBytes = enc.encode(file.name);
    const crc = crc32(file.data);
    const size = file.data.length;

    const local = concat([
      u32(0x04034b50),
      u16(20),
      u16(0),
      u16(0),
      u16(0),
      u16(0),
      u32(crc),
      u32(size),
      u32(size),
      u16(nameBytes.length),
      u16(0),
      nameBytes,
      file.data,
    ]);
    chunks.push(local);

    const cdir = concat([
      u32(0x02014b50),
      u16(20),
      u16(20),
      u16(0),
      u16(0),
      u16(0),
      u16(0),
      u32(crc),
      u32(size),
      u32(size),
      u16(nameBytes.length),
      u16(0),
      u16(0),
      u16(0),
      u16(0),
      u32(0),
      u32(offset),
      nameBytes,
    ]);
    central.push(cdir);
    offset += local.length;
  }

  const centralBytes = concat(central);
  const centralOffset = offset;
  const end = concat([
    u32(0x06054b50),
    u16(0),
    u16(0),
    u16(files.length),
    u16(files.length),
    u32(centralBytes.length),
    u32(centralOffset),
    u16(0),
  ]);

  return concat([...chunks, centralBytes, end]);
}

function concat(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

/* ------------------------------ Chart images ----------------------------- */

/** Find the first rendered <svg> inside a chart container. */
export function findChartSvg(node: HTMLElement | null): SVGSVGElement | null {
  return node?.querySelector("svg") ?? null;
}

function serialiseSvg(svg: SVGSVGElement): { xml: string; width: number; height: number } {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const rect = svg.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width || svg.clientWidth || 800));
  const height = Math.max(1, Math.round(rect.height || svg.clientHeight || 400));
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  // A solid background so the PNG isn't transparent on a slide.
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("width", "100%");
  bg.setAttribute("height", "100%");
  bg.setAttribute("fill", "#0b1020");
  clone.insertBefore(bg, clone.firstChild);
  return { xml: new XMLSerializer().serializeToString(clone), width, height };
}

/** Serialise a chart's SVG to a standalone .svg Blob. */
export function chartSvgBlob(svg: SVGSVGElement): Blob {
  const { xml } = serialiseSvg(svg);
  return new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${xml}`], { type: "image/svg+xml" });
}

/** Rasterise a chart's SVG to a PNG Blob at the given pixel scale. */
export function chartPngBlob(svg: SVGSVGElement, scale = 2): Promise<Blob> {
  const { xml, width, height } = serialiseSvg(svg);
  const svgUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(xml)}`;
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width * scale;
      canvas.height = height * scale;
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("Canvas 2D context unavailable"));
      ctx.fillStyle = "#0b1020";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("PNG encode failed"))), "image/png");
    };
    img.onerror = () => reject(new Error("Could not load chart SVG for rasterisation"));
    img.src = svgUrl;
  });
}
