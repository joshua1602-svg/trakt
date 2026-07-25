/**
 * Scene 2 — Onboard once.
 *
 * Both source schemas animate into the Onboarding Agent, which resolves them into
 * one canonical model. Every mapping row shown is a REAL mapping decision from the
 * captured onboarding contract: the source header, the canonical field it resolved
 * to, and the tier it resolved at. The counters (mapped, resolved by the client
 * contract, referred for review) are the contract's own figures.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { Frame } from "../components/Frame";
import { Card, Chevron, Eyebrow, Pill, Tick, TraktMark } from "../components/primitives";
import { C, THEME, TYPE } from "../design/tokens";
import { enter, fade, s } from "../design/motion";
import { SCHEMAS, portfolio, schemaFor } from "../data/fixtures";
import type { Caption } from "../timeline";

/** The onboarding activities, each naming what actually performs it. */
const ACTIVITIES = [
  { label: "Source profiling", detail: "type, null rate, cardinality, identifiers" },
  { label: "Field mapping", detail: "canonical model, tier by tier" },
  { label: "Schema diagnostics", detail: "unmapped and low-confidence headers" },
  { label: "Transformation rules", detail: "LTV, balance coherence, UK geography" },
  { label: "Validation configuration", detail: "canonical and business rules" },
  { label: "Approved canonical contract", detail: "applied unchanged every period" },
];

const MAPPING_ROWS_A = 5;

const SchemaColumn: React.FC<{
  portfolioKey: string;
  frame: number;
  delay: number;
  align: "left" | "right";
}> = ({ portfolioKey, frame, delay, align }) => {
  const p = portfolio(portfolioKey);
  const schema = schemaFor(portfolioKey);
  const rows = schema.columns
    .filter((c) => c.canonical_field)
    .slice(0, MAPPING_ROWS_A);
  return (
    <div style={{ width: 372, flexShrink: 0, ...enter(frame, delay, s(0.65)) }}>
      <Eyebrow style={{ textAlign: align }}>{p.display_id}</Eyebrow>
      <div
        style={{
          marginTop: 6,
          fontSize: TYPE.bodySmall,
          color: C.ink400,
          textAlign: align,
          lineHeight: 1.4,
        }}
      >
        {schema.system_of_record}
      </div>
      <div style={{ marginTop: 15, display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map((c, i) => (
          <div
            key={c.header}
            style={{
              opacity: fade(frame, delay + s(0.3) + i * s(0.1), s(0.35)),
              padding: "10px 13px",
              borderRadius: 6,
              background: "rgba(255,255,255,0.032)",
              border: `1px solid ${C.lineSoft}`,
              fontSize: TYPE.label,
              color: C.ink300,
              textAlign: align,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {c.header}
          </div>
        ))}
      </div>
    </div>
  );
};

const MappingLedger: React.FC<{ frame: number; delay: number }> = ({ frame, delay }) => {
  // Two representative decisions per portfolio: one resolved globally, one by the
  // client's approved contract, so the ledger shows both paths honestly.
  const a = schemaFor("A");
  const b = schemaFor("B");
  const pick = (schema: typeof a, resolution: string, n: number) =>
    schema.columns.filter((c) => c.canonical_field && c.resolution === resolution).slice(0, n);
  const rows = [
    ...pick(a, "contract", 2).map((c) => ({ ...c, book: portfolio("A").short_label })),
    ...pick(b, "contract", 2).map((c) => ({ ...c, book: portfolio("B").short_label })),
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
      {rows.map((r, i) => (
        <div
          key={`${r.book}-${r.header}`}
          style={{
            opacity: fade(frame, delay + i * s(0.13), s(0.4)),
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "9px 12px",
            borderRadius: 6,
            background: "rgba(255,255,255,0.025)",
            border: `1px solid ${C.lineSoft}`,
          }}
        >
          <span
            style={{
              width: 88,
              flexShrink: 0,
              fontSize: TYPE.micro,
              color: C.ink500,
              letterSpacing: "0.05em",
              textTransform: "uppercase",
            }}
          >
            {r.book}
          </span>
          <span
            style={{
              width: 196,
              flexShrink: 0,
              fontSize: TYPE.label,
              color: C.ink300,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {r.header}
          </span>
          <Chevron size={13} colour={C.peri400} />
          <span
            style={{
              flex: 1,
              minWidth: 0,
              fontSize: TYPE.label,
              fontFamily: "inherit",
              color: C.ink100,
              fontWeight: 500,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {r.canonical_field}
          </span>
          <Tick size={12} />
        </div>
      ))}
    </div>
  );
};

export const Scene2Onboard: React.FC<{ captions: Caption[] }> = ({ captions }) => {
  const frame = useCurrentFrame();
  const schemaA = SCHEMAS[portfolio("A").schema_name];
  const schemaB = SCHEMAS[portfolio("B").schema_name];
  const contractCountA = schemaA.columns.filter((c) => c.resolution === "contract").length;
  const contractCountB = schemaB.columns.filter((c) => c.resolution === "contract").length;
  const mappedA = schemaA.columns.filter((c) => c.canonical_field).length;
  const mappedB = schemaB.columns.filter((c) => c.canonical_field).length;
  const unmapped =
    schemaA.columns.filter((c) => !c.canonical_field).length +
    schemaB.columns.filter((c) => !c.canonical_field).length;

  return (
    <Frame captions={captions} chapter="Stage 1 · Onboard once">
      <div style={{ display: "flex", flexDirection: "column", gap: 20, flex: 1, minHeight: 0 }}>
        {/* Two schemas converging on the agent. */}
        <div style={{ display: "flex", alignItems: "flex-start", gap: 26 }}>
          <SchemaColumn portfolioKey="A" frame={frame} delay={s(0.3)} align="left" />

          <div
            style={{
              flex: 1,
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 14,
              paddingTop: 22,
              ...enter(frame, s(1.0), s(0.7)),
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 13,
                padding: "13px 22px",
                borderRadius: 999,
                background: "rgba(145,157,209,0.10)",
                border: `1px solid ${C.navy500}`,
              }}
            >
              <TraktMark size={30} />
              <span style={{ fontSize: TYPE.subheading, fontWeight: 600, letterSpacing: "-0.01em" }}>
                Onboarding Agent
              </span>
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 9,
                width: "100%",
              }}
            >
              {ACTIVITIES.map((a, i) => (
                <div
                  key={a.label}
                  style={{
                    opacity: fade(frame, s(2.2) + i * s(0.42), s(0.45)),
                    padding: "10px 13px",
                    borderRadius: 6,
                    border: `1px solid ${
                      i === ACTIVITIES.length - 1 ? "rgba(46,125,91,0.42)" : C.lineSoft
                    }`,
                    background:
                      i === ACTIVITIES.length - 1
                        ? "rgba(46,125,91,0.10)"
                        : "rgba(255,255,255,0.022)",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    {i === ACTIVITIES.length - 1 ? <Tick size={12} /> : null}
                    <span
                      style={{
                        fontSize: TYPE.label,
                        fontWeight: 600,
                        color: i === ACTIVITIES.length - 1 ? C.ink100 : C.ink300,
                      }}
                    >
                      {a.label}
                    </span>
                  </div>
                  <div style={{ marginTop: 3, fontSize: TYPE.micro, color: C.ink500 }}>
                    {a.detail}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <SchemaColumn portfolioKey="B" frame={frame} delay={s(0.6)} align="right" />
        </div>

        {/* The resulting contract: real mapping decisions and real counters. */}
        <Card
          style={{ opacity: fade(frame, s(5.4), s(0.8)) }}
          accent={THEME.positive}
          padding={22}
        >
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              justifyContent: "space-between",
              marginBottom: 15,
            }}
          >
            <div>
              <Eyebrow>Approved onboarding contract</Eyebrow>
              <div
                style={{
                  marginTop: 5,
                  fontSize: TYPE.body,
                  fontWeight: 600,
                  color: C.ink100,
                }}
              >
                Source header → Trakt canonical field
              </div>
            </div>
            <div style={{ display: "flex", gap: 10 }}>
              <Pill tone="positive">
                <Tick size={11} />
                {mappedA + mappedB} fields mapped
              </Pill>
              <Pill tone="accent">
                {contractCountA + contractCountB} client-specific decisions
              </Pill>
              <Pill tone="warning">{unmapped} referred for review</Pill>
            </div>
          </div>
          <MappingLedger frame={frame} delay={s(6.0)} />
        </Card>

        {/* What the contract establishes for every future period. */}
        <div
          style={{
            marginTop: "auto",
            display: "flex",
            gap: 14,
            opacity: fade(frame, s(13.2), s(0.9)),
          }}
        >
          {[
            {
              title: "Transformations",
              items: [
                "Balance coherence",
                "Current and original LTV",
                "Postcode → UK ITL3 geography",
                "Source-portfolio provenance stamp",
              ],
            },
            {
              title: "Validation",
              items: [
                "Gate 2 · canonical conformance",
                "Gate 2.5 · field and value lineage",
                "Gate 3 · cross-field business rules",
                "Gate 3b · exception aggregation",
              ],
            },
            {
              title: "Recurring execution",
              items: [
                "Applied unchanged every period",
                "No mapping decision repeated",
                "Monthly, on receipt of the extract",
                "Assembled into the platform view",
              ],
            },
          ].map((group, gi) => (
            <div
              key={group.title}
              style={{
                flex: 1,
                minWidth: 0,
                padding: "16px 18px",
                borderRadius: 8,
                border: `1px solid ${C.line}`,
                background: "rgba(255,255,255,0.02)",
              }}
            >
              <Eyebrow color={C.peri300}>{group.title}</Eyebrow>
              <div style={{ marginTop: 11, display: "flex", flexDirection: "column", gap: 7 }}>
                {group.items.map((item, ii) => (
                  <div
                    key={item}
                    style={{
                      display: "flex",
                      gap: 9,
                      alignItems: "center",
                      fontSize: TYPE.label,
                      color: C.ink300,
                      opacity: fade(frame, s(13.4) + gi * s(0.2) + ii * s(0.08), s(0.4)),
                    }}
                  >
                    <Tick size={11} />
                    <span
                      style={{
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {item}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Frame>
  );
};
