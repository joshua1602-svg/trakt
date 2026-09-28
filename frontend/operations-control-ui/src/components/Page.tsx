import type { ReactNode } from "react";
import clsx from "clsx";

/** How wide the content column runs.
 *
 *  `reading` is the default, and is right for prose and forms: a measure a
 *  person can actually read a sentence across. Every screen has used it.
 *
 *  `wide` is for an operational screen whose subject is a TABLE. A mapping
 *  table is not prose — it is a grid an operator scans across, one row per
 *  source column, and a reading measure is the wrong container for it. Capped
 *  at the reading width the mapping table had 528px to render five columns in
 *  (1024, less the page gutters, less the 24rem status rail, less the panel
 *  padding), so every row wrapped onto two lines and a hundred-column tape
 *  became two hundred lines of zig-zag.
 *
 *  It is a cap, not a width: the page still centres, and on an ultrawide
 *  display the conversation and the prose beside the table do not stretch to
 *  an unreadable measure.
 */
export type PageWidth = "reading" | "wide";

const WIDTHS: Record<PageWidth, string> = {
  reading: "max-w-5xl",
  wide: "max-w-[100rem]",
};

/** Shared page wrapper: calm heading, generous whitespace. */
export function Page({
  title,
  subtitle,
  actions,
  width = "reading",
  children,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  width?: PageWidth;
  children: ReactNode;
}) {
  return (
    <div className={clsx("mx-auto w-full px-6 py-10", WIDTHS[width])}>
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-stone-900">{title}</h1>
          {subtitle && <p className="mt-1 text-sm text-stone-500">{subtitle}</p>}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </header>
      {children}
    </div>
  );
}
