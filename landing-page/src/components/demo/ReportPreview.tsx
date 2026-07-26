"use client";

import { useState } from "react";

import { ArtifactBlock } from "@/components/demo/Artifacts";
import { Badge, buttonStyles, cx } from "@/components/ui";
import type { DemoReportResponse } from "@/types/demo";

/**
 * In-page preview of a synthetic report.
 *
 * The pages are rendered from the deterministic engine output carried in the
 * demo pack. There is no document, no download and no storage URL — a public
 * visitor can look at the report, not take it.
 */
export function ReportPreview({
  payload,
  onFollowUp,
  onReport,
  reportIds,
}: {
  payload: DemoReportResponse;
  onFollowUp: (id: string, label: string) => void;
  onReport: (id: string, label: string) => void;
  reportIds: string[];
}) {
  const [pageIndex, setPageIndex] = useState(0);
  const pages = payload.pages ?? [];
  const page = pages[pageIndex];

  if (payload.status === "limit_reached") {
    return (
      <article className="animate-rise rounded-xl border border-peri-500/40 bg-peri-400/[0.06] p-4 sm:p-5">
        <p className="text-sm leading-relaxed text-ink-100">{payload.answer}</p>
        <a href="#book-a-demo" className={cx(buttonStyles.primary, "mt-4")}>
          Book a portfolio walkthrough
        </a>
      </article>
    );
  }

  if (!page) {
    return (
      <article className="animate-rise rounded-xl border border-line bg-navy-950/60 p-4">
        <p className="text-sm text-ink-100">
          {payload.answer ?? "This preview is unavailable."}
        </p>
      </article>
    );
  }

  return (
    <article className="animate-rise rounded-xl border border-line bg-navy-950/60 p-4 sm:p-5">
      <p className="text-sm leading-relaxed text-ink-100">{payload.answer}</p>

      <section
        aria-label={`${payload.documentTitle} preview`}
        className="mt-4 rounded-lg border border-line bg-navy-900/70"
      >
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
          <div>
            <p className="text-sm font-semibold text-ink-100">{payload.documentTitle}</p>
            <p className="mt-0.5 text-[11px] text-ink-400">{payload.documentSubtitle}</p>
          </div>
          <Badge tone="synthetic">Synthetic preview</Badge>
        </header>

        {/* A fixed minimum height keeps page switching from shifting the layout. */}
        <div className="min-h-[300px] px-4 py-4">
          <h4 className="text-[13px] font-semibold uppercase tracking-wider text-peri-300">
            {page.title}
          </h4>
          {page.subtitle ? (
            <p className="mt-1 text-[11px] text-ink-400">{page.subtitle}</p>
          ) : null}

          <div className="mt-4 space-y-5">
            {page.blocks.map((block, index) => (
              <ArtifactBlock key={index} artifact={block} />
            ))}
          </div>

          {page.note ? (
            <p className="mt-5 border-t border-line-soft pt-3 text-[11px] leading-relaxed text-ink-500">
              {page.note}
            </p>
          ) : null}
        </div>

        <nav
          aria-label="Report pages"
          className="flex flex-wrap items-center gap-2 border-t border-line px-4 py-3"
        >
          {pages.map((candidate, index) => (
            <button
              key={candidate.title}
              type="button"
              aria-current={index === pageIndex ? "page" : undefined}
              onClick={() => setPageIndex(index)}
              className={cx(
                "rounded-md px-2.5 py-1 text-[11px] transition-colors",
                index === pageIndex
                  ? "bg-peri-400 font-semibold text-navy-950"
                  : "border border-line text-ink-400 hover:text-ink-100",
              )}
            >
              <span className="sr-only">Page {index + 1}: </span>
              {index + 1}
            </button>
          ))}
          <span className="ml-1 text-[11px] text-ink-500">
            Page {pageIndex + 1} of {pages.length} · preview only, no document is produced
          </span>
        </nav>
      </section>

      {payload.followUps?.length ? (
        <ul className="mt-4 flex flex-wrap gap-2">
          {payload.followUps.map((followUp) => (
            <li key={followUp.id}>
              <button
                type="button"
                onClick={() =>
                  reportIds.includes(followUp.id)
                    ? onReport(followUp.id, followUp.label)
                    : onFollowUp(followUp.id, followUp.label)
                }
                className="rounded-full border border-line px-3 py-1 text-xs text-ink-300 transition-colors hover:border-peri-500 hover:text-ink-100"
              >
                {followUp.label}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}
