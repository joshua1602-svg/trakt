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
 * Now it sits under the demo frame as live text. Not small print: it keeps
 * heading-scale type and a rule above it, so it reads as the conclusion the
 * demo has just earned rather than as a caption. The prompts stay real
 * buttons — pressing one starts the demo above and asks it, so the visitor
 * watches Trakt decline rather than being told that it does.
 */

/** Fired at the query demo, which starts itself and asks the question. */
export const ASK_EVENT = "trakt:ask";

export function RefusalSection({ meta }: { meta: DemoMetaResponse }) {
  if (meta.exampleUnsupported.length === 0) return null;

  return (
    <div className="mt-10 border-t border-line pt-8">
      <h3 className="text-balance text-xl font-semibold tracking-tight text-ink-100 sm:text-2xl">
        Trakt declines what it cannot derive.
      </h3>
      <p className="mt-3 max-w-[72ch] text-[15px] leading-relaxed text-ink-300">
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
              className="rounded-full border border-mint-400/35 bg-navy-850 px-4 py-2 text-left text-sm text-ink-100 transition-colors hover:border-mint-400/70"
            >
              {example.label}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
