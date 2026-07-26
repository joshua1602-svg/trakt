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
 * The one arriving cut the demonstration does not itself carry.
 *
 * Four of the five inputs S2 shows are real: three are the source schemas the generator
 * actually writes (`src/data/fixtures.ts` derives them from `SCHEMAS`), and the risk
 * limit schedule is the Schedule 8 document the concentration monitor genuinely reads.
 * A warehouse facility schedule is part of this buyer's month-end and is not in the
 * synthetic set, so it is declared here as stated copy rather than dressed up as data.
 */
export const STATED_ARRIVAL = {
  title: "Warehouse facility schedule",
  format: "XLSX",
  owner: "funding counterparty",
  frequency: "monthly",
};

/** Plain-English use lines for the three channels in S4, keyed by channel label. */
export const CHANNEL_USE: Record<string, string> = {
  "Managed service": "You never log in.",
  "Microsoft 365 Copilot": "Ask from where you already work.",
  "MI Agent workspace": "Open it when you need to drill in.",
};
