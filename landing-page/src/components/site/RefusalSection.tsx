"use client";

import { SectionHeading } from "@/components/ui";
import type { DemoMetaResponse } from "@/types/demo";

/**
 * The refusal claim, as its own section.
 *
 * It used to sit inside the demo card, which put the page's strongest trust
 * claim behind a play control — visible only as text in a still. The prompts
 * stay real buttons: pressing one starts the query demo above and asks it, so
 * the visitor sees Trakt decline rather than being told that it does.
 */

/** Fired at the query demo, which starts itself and asks the question. */
export const ASK_EVENT = "trakt:ask";

export function RefusalSection({ meta }: { meta: DemoMetaResponse }) {
  if (meta.exampleUnsupported.length === 0) return null;

  return (
    <>
      <SectionHeading
        id="refusal"
        eyebrow="Boundaries"
        title="Trakt declines what it cannot derive."
      />
      <p className="mt-4 max-w-[72ch] text-[15px] leading-relaxed text-ink-300">
        Ask either of these and watch it refuse, with the reason.
      </p>
      <ul className="mt-6 flex flex-wrap gap-2.5">
        {meta.exampleUnsupported.map((example) => (
          <li key={example.id}>
            <button
              type="button"
              onClick={() => {
                window.dispatchEvent(
                  new CustomEvent(ASK_EVENT, { detail: { id: example.id, label: example.label } }),
                );
              }}
              className="rounded-full border border-mint-400/35 bg-navy-850 px-4 py-2 text-left text-sm text-ink-100 transition-colors hover:border-mint-400/70"
            >
              {example.label}
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}
