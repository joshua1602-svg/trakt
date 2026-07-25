/**
 * UK English, pounds sterling, British number formatting.
 *
 * These are PRESENTATION helpers only — they format a value the fixtures already
 * fixed. They never round in a way that changes a stated figure: the money
 * formatter keeps one decimal at millions (£18.1m), which is the precision the
 * product's own answers use, and the assertion gate compares the two surfaces on
 * exactly those rendered figures.
 */

const NUMBER = new Intl.NumberFormat("en-GB");
const NUMBER_1DP = new Intl.NumberFormat("en-GB", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const NUMBER_2DP = new Intl.NumberFormat("en-GB", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

/** 11,035 */
export const count = (value: number | null | undefined): string =>
  value == null ? "—" : NUMBER.format(Math.round(value));

/**
 * £1.96bn / £18.1m / £516.2m / £178k — the same abbreviation ladder the product's
 * currency formatter uses (mi_agent_api/currency.py: bn / m / k).
 */
export const money = (value: number | null | undefined): string => {
  if (value == null) return "—";
  const abs = Math.abs(value);
  const sign = value < 0 ? "−" : "";
  if (abs >= 1e9) return `${sign}£${NUMBER_2DP.format(abs / 1e9)}bn`;
  if (abs >= 1e6) return `${sign}£${NUMBER_1DP.format(abs / 1e6)}m`;
  if (abs >= 1e3) return `${sign}£${NUMBER.format(Math.round(abs / 1e3))}k`;
  return `${sign}£${NUMBER.format(Math.round(abs))}`;
};

/** £18.1m with an explicit sign, for a movement. */
export const signedMoney = (value: number | null | undefined): string => {
  if (value == null) return "—";
  const formatted = money(Math.abs(value));
  return `${value < 0 ? "−" : "+"}${formatted}`;
};

/** 43.2% */
export const percent = (
  value: number | null | undefined,
  decimals = 1,
): string => {
  if (value == null) return "—";
  const fmt = new Intl.NumberFormat("en-GB", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return `${fmt.format(value)}%`;
};

/** +0.28pp */
export const signedPoints = (
  value: number | null | undefined,
  decimals = 2,
): string => {
  if (value == null) return "—";
  const fmt = new Intl.NumberFormat("en-GB", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return `${value >= 0 ? "+" : "−"}${fmt.format(Math.abs(value))}pp`;
};

/** 71.4 years */
export const years = (value: number | null | undefined, decimals = 1): string => {
  if (value == null) return "—";
  const fmt = new Intl.NumberFormat("en-GB", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return `${fmt.format(value)} years`;
};

/** 30 June 2026 */
export const longDate = (iso: string | null | undefined): string => {
  if (!iso) return "—";
  const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(d);
};

/** 30 Jun 2026 */
export const shortDate = (iso: string | null | undefined): string => {
  if (!iso) return "—";
  const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(d);
};

/** Jun 2026, from a YYYY-MM period key. */
export const periodLabel = (period: string | null | undefined): string => {
  if (!period) return "—";
  const d = new Date(`${period}-01T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return period;
  return new Intl.DateTimeFormat("en-GB", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(d);
};

/** 1.2 MB */
export const fileSize = (bytes: number | null | undefined): string => {
  if (bytes == null) return "—";
  if (bytes >= 1e6) return `${NUMBER_1DP.format(bytes / 1e6)} MB`;
  if (bytes >= 1e3) return `${NUMBER.format(Math.round(bytes / 1e3))} KB`;
  return `${NUMBER.format(bytes)} B`;
};

/** 43% share, from a 0..1 fraction. */
export const share = (fraction: number | null | undefined, decimals = 0): string =>
  fraction == null ? "—" : percent(fraction * 100, decimals);

/**
 * Keep a signed money token from breaking across a line.
 *
 * Chrome will happily wrap between "+" and "£", which renders an answer as
 * "contributed approximately +" / "£11.5m" across two lines. This inserts a word
 * joiner (U+2060) after the sign so the token stays together. Display only — no
 * digit, sign or word is altered, and the resulting string still matches the
 * figure-comparison regexes the assertion gate uses.
 */
export const noBreakSigns = (text: string): string =>
  text.replace(/([+\u2212-])(?=[£$€]|\d)/g, "$1\u2060");
