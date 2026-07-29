/** Date helpers — friendly, date-only formatting for operators. */

export function formatDate(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

/** Reporting periods arrive as YYYY-MM-DD (period end) or YYYY-MM. */
export function formatPeriod(value: string | null | undefined): string {
  if (!value) return "";
  const monthMatch = /^(\d{4})-(\d{2})/.exec(value);
  if (monthMatch) {
    const date = new Date(Number(monthMatch[1]), Number(monthMatch[2]) - 1, 1);
    return date.toLocaleDateString("en-GB", { month: "long", year: "numeric" });
  }
  return value;
}

/** "mapping" -> "Mapping", "new_portfolio" -> "New portfolio". */
export function humanize(value: string): string {
  if (!value) return "";
  const spaced = value.replace(/[_-]+/g, " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
