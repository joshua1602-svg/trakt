/**
 * Stated claims — copy the business stands behind, not figures the pipeline produced.
 *
 * Everything the film shows falls into one of two categories, and they must never be
 * confused with one another:
 *
 *   MEASURED  — read from `public/fixtures/*.json`, which the demonstration run wrote.
 *               Accessed through `src/data/fixtures.ts`, set in IBM Plex Mono, and
 *               always carrying a provenance rule that names where it came from.
 *
 *   STATED    — commercial claims about the product and its market. They live here,
 *               in one file, so that anyone auditing the film can see the complete
 *               list of things it asserts without a fixture behind them.
 *
 * A number in this file is the business's to defend on the call the film generates.
 * A number in a fixture is the pipeline's. Do not move anything between the two.
 */

/**
 * Onboarding elapsed time: tape received to governed output.
 *
 * Rendered as an `HH:MM` clock. Deliberately short of the 48-hour threshold the claim
 * states — a figure with visible headroom is more credible than one that lands exactly
 * on its own limit, and it is the number that has to survive the first question about
 * it.
 */
export const ONBOARDING_HOURS = 41 + 20 / 60;

/** The threshold the claim states, in hours. The clock must come in under it. */
export const ONBOARDING_CLAIM_HOURS = 48;

/**
 * What Trakt replaces, in the buyer's own terms. Every other line in the film is
 * about capability; this is the only one about cost, and it is the reason a COO at a
 * twelve-person lender keeps watching.
 */
export const MONTH_END_COST = "Five days of month-end. Every month. For every portfolio.";

/** S1's opening line: the buyer, before any artefact is named. */
export const OPENING_LINE = "You bought a back book. It didn't come with your data model.";

/**
 * The five artefacts a specialist lender's month-end actually runs on.
 *
 * STATED, not measured, and all five of them — which is a deliberate change from the
 * previous version, where four rows were derived from the demonstration's own source
 * schemas. The funnel is making a claim about the buyer's world, not about this
 * synthetic portfolio: these are the artefact TYPES that arrive, in the buyer's own
 * vocabulary, and dressing three of them up as fixture-derived would be claiming
 * measurement for something that is really industry knowledge.
 *
 * The demonstration does carry three of them for real — the loan tape and the
 * collateral tape as `origination_extract` / `servicer_account_extract` /
 * `trustee_deal_extract`, and the covenant schedule as the Schedule 8 limits the
 * concentration monitor reads. That is why the claim is safe to make. It is not why it
 * is true, so it lives here.
 *
 * Three formats and three frequencies, not five and three: these five artefacts carry
 * CSV, XLSX and PDF between them. The heterogeneity the beat is showing is real without
 * inventing two more file types to reach a number.
 */
export const ARRIVING_ARTEFACTS = [
  { title: "Loan tape", format: "CSV", frequency: "monthly" },
  { title: "Collateral tape", format: "CSV", frequency: "monthly" },
  { title: "Cash flows", format: "XLSX", frequency: "monthly" },
  { title: "Warehouse agreement", format: "PDF", frequency: "on change" },
  { title: "Covenant schedule", format: "XLSX", frequency: "quarterly" },
] as const;

/** The line the funnel resolves into. */
export const FUNNEL_CLAIM = "Every artifact your portfolio runs on. One governed dataset.";

/**
 * The Microsoft product name, exactly as Microsoft's trademark guidelines require it:
 * "Microsoft" first, the product name unaltered and unabbreviated.
 *
 * Microsoft Legal states that "our logos, app and product icons, illustrations,
 * photographs, videos, and designs can never be used without an express license", which
 * this project does not hold — so S4's panel carries this wordmark, set in the body face,
 * and no app icon. See the header of `src/scenes/S4Omnichannel.tsx`.
 */
export const MICROSOFT_CHANNEL = "Microsoft 365 Copilot";

/** Attribution for the wordmark above. Rendered in S4's result band. */
export const MS_TRADEMARK_NOTICE =
  "MICROSOFT AND MICROSOFT 365 COPILOT ARE TRADEMARKS OF THE MICROSOFT GROUP OF COMPANIES";

/**
 * The question S4's Copilot panel shows.
 *
 * Stated, not measured: it is a person typing, so there is no fixture behind it. The
 * ANSWER beneath it is condensed from the recorded turn in `demo_metrics.json`, and a
 * test holds the condensation to what the agent actually said.
 */
export const COPILOT_ASK = "What changed versus last month?";

/** Plain-English use lines for the three channels in S4, keyed by channel label. */
export const CHANNEL_USE: Record<string, string> = {
  "Managed service": "You never log in.",
  [MICROSOFT_CHANNEL]: "Ask from where you already work.",
  "MI Agent workspace": "Open it when you need to drill in.",
};
