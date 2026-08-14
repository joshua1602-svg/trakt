import { Reveal } from "@/components/site/Reveal";
import { Card, SectionHeading, cx } from "@/components/ui";

/**
 * Capability — one portfolio, every workflow.
 *
 * The Platform section says "Build the portfolio once. Use it everywhere."
 * Nothing on the page said what "everywhere" meant. This does, and it does it
 * as a matrix rather than prose because seventeen items in five groups is a
 * shape, not an argument — the reader should be able to find their own job in
 * it in one pass.
 *
 * **No item descriptions and no paragraph.** A line under each item would make
 * this a fourth explanatory section; the items are nouns a lender already
 * knows, and the sections around this one carry the depth (§2 the query demo,
 * §4 the controls demo, §6c the agents). Overview here, depth there.
 *
 * **It replaces the Delivery Model section.** That section's four tiles said
 * what this column's five items say, at a section's cost. Its headline — "From
 * a managed service to your own agents" — is not rehomed: the ladder it named
 * is carried by the DELIVERY items themselves, managed service at one end and
 * A2A at the other. Its body line moved to Risk & Controls, where the finding
 * being pushed is actually produced.
 *
 * **Status is marked, not implied.** Fifteen of the seventeen items ship today;
 * two do not, and they carry the page's existing ROADMAP convention — grey
 * caps, the same treatment the deleted Delivery tiles used for exactly this.
 * Grey shading alone was considered and rejected: a reader who misses the
 * shade reads seventeen shipped items. Every item's grounding is recorded in
 * `docs/content-map.md`.
 *
 * **Deliberately excluded: a generic "Forecasting" item.** Trakt forecasts the
 * pipeline (`analytics/pipeline_expected_funding.py`,
 * `analytics/pipeline_forward_risk.py`), not the funded book, and an unqualified
 * "Forecasting" in a grid of shipped capabilities claims the second. "Pipeline"
 * is the honest word and it is the first item in the first group.
 */

type Item = { readonly label: string; readonly roadmap?: true };

const GROUPS: readonly { readonly label: string; readonly items: readonly Item[] }[] = [
  {
    label: "Portfolio",
    items: [
      { label: "Pipeline" },
      { label: "Total AUM" },
      { label: "Portfolio / asset / SPV views" },
    ],
  },
  {
    label: "Control",
    items: [
      { label: "Concentration limits" },
      { label: "Proactive risk watchlist" },
      { label: "Governance & lineage" },
    ],
  },
  {
    label: "Report",
    items: [
      { label: "Management MI" },
      { label: "Warehouse reporting" },
      // Never shortened to "Securitisation": it has to stay distinguishable
      // from "Securitisation Readiness" two columns to the right, which is a
      // different thing done by a different component.
      { label: "Securitisation reporting" },
    ],
  },
  {
    label: "Investigate",
    items: [
      { label: "MI Query" },
      { label: "Securitisation Readiness" },
      { label: "Portfolio Acquisition DD", roadmap: true },
    ],
  },
  {
    // Five items rather than four, and "Managed service" first. Without it the
    // page loses its primary commercial delivery mode entirely when the
    // Delivery Model section goes — and the ladder loses its bottom rung.
    label: "Delivery",
    items: [
      { label: "Managed service" },
      { label: "Trakt" },
      { label: "Microsoft Copilot" },
      { label: "Microsoft Teams" },
      { label: "Enterprise agent A2A", roadmap: true },
    ],
  },
];

export function Capability() {
  return (
    <>
      <Reveal>
        <SectionHeading
          id="capability"
          eyebrow="Capability"
          title="One portfolio. Every workflow."
        />
      </Reveal>

      {/* Five cards is the number that produced the same defect twice in the
          governance grid — one card alone on the last row, which reads as a
          rendering fault rather than a wrap. Resolved here the same way at lg
          (3 + 2 is a wrap, not an orphan) and one step earlier at sm, where the
          fifth card spans both columns instead of sitting alone. It is the card
          with five items, so it earns the width rather than merely filling it.
          The row-occupancy guard measures the rendered boxes at five widths. */}
      <ul
        data-grid="capability"
        className="mt-9 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5"
      >
        {GROUPS.map((group, index) => (
          <li
            key={group.label}
            className={cx(index === GROUPS.length - 1 && "sm:col-span-2 lg:col-span-1")}
          >
            <Reveal delay={index * 60} className="h-full">
              <Card className="h-full">
                <h3 className="text-xs font-semibold uppercase tracking-[0.16em] text-peri-400">
                  {group.label}
                </h3>
                <ul className="mt-4 space-y-2">
                  {group.items.map((item) => (
                    <li
                      key={item.label}
                      className={cx(
                        "text-sm leading-snug",
                        item.roadmap ? "text-ink-500" : "text-ink-300",
                      )}
                    >
                      <span>{item.label}</span>
                      {item.roadmap ? (
                        <>
                          {/* Grey caps, the page's existing roadmap
                              convention — carried over from the Delivery tiles
                              that used to make this distinction. */}
                          <span className="ml-2 align-[0.1em] text-[10px] font-semibold uppercase tracking-wider text-ink-500">
                            Roadmap
                          </span>
                          {/* Machine-readable too, so the status survives a
                              reader who cannot see the label and a future edit
                              that drops it. */}
                          <span className="sr-only"> (roadmap)</span>
                        </>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </Card>
            </Reveal>
          </li>
        ))}
      </ul>
    </>
  );
}
