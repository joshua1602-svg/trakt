"use client";

import type { DemoMetaResponse } from "@/types/demo";

/**
 * The refusal claim, directly beneath the query demo it belongs to.
 *
 * It has moved twice. It began inside the demo card, which put the page's
 * strongest trust claim behind a play control — visible only as text in a
 * still. It then became its own section, which over-promoted a sub-claim of
 * the demo into a destination, and put it in adjacent territory to the
 * agent-to-agent section's "Agents don't calculate the portfolio."
 *
 * Now it sits under the demo frame as live text, and the spacing has to do
 * two contradictory things at once: keep it subordinate to the demo without
 * demoting it to small print. At full section spacing it read as a section
 * that had lost its eyebrow — every other block on the page has one. So it
 * takes the demo figure's own `max-w-4xl` and a soft rule at half the gap,
 * which ties it to the frame above; the type stays at heading scale, which is
 * what stops it becoming a caption. The prompts are real buttons — pressing
 * one starts the demo above and asks it, so the visitor watches Trakt decline
 * rather than being told that it does.
 */

/** Fired at the query demo, which starts itself and asks the question. */
export const ASK_EVENT = "trakt:ask";

export function RefusalSection({ meta }: { meta: DemoMetaResponse }) {
  if (meta.exampleUnsupported.length === 0) return null;

  return (
    <div className="mt-6 max-w-4xl border-t border-line-soft pt-6">
      <h3 className="text-balance text-headline font-semibold text-ink-100">
        Trakt declines what it cannot derive.
      </h3>
      <p className="mt-3 max-w-[72ch] text-body leading-relaxed text-ink-300">
        Ask either of these and watch it refuse, with the reason.
      </p>
      <ul className="mt-5 flex flex-wrap gap-2.5">
        {meta.exampleUnsupported.map((example) => (
          <li key={example.id}>
            <button
              type="button"
              onClick={() => {
                window.dispatchEvent(
                  new CustomEvent(ASK_EVENT, { detail: { id: example.id, label: example.label } }),
                );
              }}
              className="rounded-full border border-line bg-navy-850 px-4 py-2 text-left text-body text-ink-100 transition-colors hover:border-peri-500"
            >
              {example.label}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
