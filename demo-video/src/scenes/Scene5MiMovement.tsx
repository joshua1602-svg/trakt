/**
 * Scene 5 — Follow-up comparison.
 *
 * The same conversation, now across reporting periods. The user asks "What has
 * changed versus the prior month?" and the answer is the verbatim `/mi/query`
 * response from the governed period-movement route.
 *
 * The dashboard beside it switches to the month-on-month view: the KPI tiles carry
 * their deltas, and the regional bar list becomes the signed contribution chart
 * whose values sum exactly to the consolidated movement (the assertion gate proves
 * the residual is nil). The South East is highlighted because the fixtures say it
 * is the primary contributor — not because the scene decided so.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { Frame } from "../components/Frame";
import { DataTable, Eyebrow, KpiTile, SignedBarList } from "../components/primitives";
import {
  AgentMessage,
  AppWindow,
  ChatColumn,
  DashboardPanel,
  MiAgentHeader,
  Thinking,
  UserMessage,
} from "../components/MiAgentShell";
import { C, TYPE } from "../design/tokens";
import { enter, fade, s, typed, typedDuration } from "../design/motion";
import {
  CURRENT_PERIOD,
  MOVEMENT,
  PRIMARY_REGION,
  PRIOR_PERIOD,
  contribution,
  miTurn,
} from "../data/fixtures";
import {
  count,
  money,
  noBreakSigns,
  percent,
  periodLabel,
  signedMoney,
  signedPoints,
  years,
} from "../design/format";
import type { Caption } from "../timeline";

const TURN = miTurn(1);

const T = {
  windowIn: 0.1,
  priorTurn: 0.6,
  question: 2.0,
  thinking: 3.6,
  answer: 4.4,
} as const;

export const Scene5MiMovement: React.FC<{ captions: Caption[] }> = ({ captions }) => {
  const frame = useCurrentFrame();
  const answerStart = s(T.answer);
  const answerText = noBreakSigns(typed(TURN.answer, frame, answerStart, 3.2));
  const answerDone = frame > answerStart + typedDuration(TURN.answer, 3.2);
  const tablesAt = answerStart + typedDuration(TURN.answer, 3.2) + s(0.3);

  const d = MOVEMENT.delta;
  const cur = MOVEMENT.current;
  const a = contribution("A");
  const b = contribution("B");

  // Top regional movers, largest positive first — straight from the bridge.
  const contributions = MOVEMENT.regionContributions.slice(0, 9);

  return (
    <Frame captions={captions} chapter="Trakt MI Agent · month on month">
      <AppWindow style={{ ...enter(frame, s(T.windowIn), s(0.5), 8) }}>
        <MiAgentHeader
          portfolioValue="All Portfolios"
          reportingDate={CURRENT_PERIOD.reportingDate}
        />
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <DashboardPanel
            reportingDate={CURRENT_PERIOD.reportingDate}
            title={`Movement · ${periodLabel(PRIOR_PERIOD.period)} → ${periodLabel(
              CURRENT_PERIOD.period,
            )}`}
          >
            {/* KPI tiles with their month-on-month deltas. */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 12,
                opacity: fade(frame, s(0.7), s(0.7)),
              }}
            >
              <KpiTile
                label="Funded balance movement"
                value={signedMoney(d.funded_balance)}
                emphasis
                deltaPositive={(d.funded_balance ?? 0) >= 0}
              />
              <KpiTile
                label={`Funded balance · ${periodLabel(CURRENT_PERIOD.period)}`}
                value={money(cur.funded_balance)}
              />
              <KpiTile
                label="Loans funded"
                value={count(cur.loan_count)}
                delta={`${(d.loan_count ?? 0) >= 0 ? "+" : "−"}${count(
                  Math.abs(d.loan_count ?? 0),
                )}`}
                deltaPositive={(d.loan_count ?? 0) >= 0}
              />
              <KpiTile
                label="WA current LTV"
                value={percent(cur.wa_ltv_points)}
                delta={signedPoints(d.wa_ltv_points)}
                deltaPositive={(d.wa_ltv_points ?? 0) >= 0}
              />
              <KpiTile
                label="Avg borrower age"
                value={years(cur.avg_borrower_age)}
                delta={`${(d.avg_borrower_age ?? 0) >= 0 ? "+" : "−"}${Math.abs(
                  d.avg_borrower_age ?? 0,
                ).toFixed(2)}`}
                deltaPositive={(d.avg_borrower_age ?? 0) >= 0}
              />
              <KpiTile
                label="WA interest rate"
                value={percent(cur.wa_interest_rate, 2)}
                // A delta that rounds to 0.00pp is not shown: "+0.00pp" reads as
                // a measurement failure rather than as "unchanged".
                delta={
                  Math.abs(d.wa_interest_rate ?? 0) >= 0.005
                    ? signedPoints(d.wa_interest_rate)
                    : undefined
                }
                deltaPositive={(d.wa_interest_rate ?? 0) >= 0}
              />
            </div>

            <div style={{ display: "flex", gap: 18, flex: 1, minHeight: 0 }}>
              {/* Signed regional contribution — sums to the consolidated movement. */}
              <div style={{ flex: 1.3, minWidth: 0, display: "flex", flexDirection: "column" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <Eyebrow>Contribution to the movement, by region</Eyebrow>
                  <span style={{ fontSize: TYPE.micro, color: C.ink500 }}>
                    sums to {signedMoney(d.funded_balance)}
                  </span>
                </div>
                <div style={{ marginTop: 14, flex: 1 }}>
                  <SignedBarList
                    data={contributions.map((c) => ({ label: c.region, value: c.delta }))}
                    formatValue={signedMoney}
                    progress={fade(frame, s(1.5), s(1.3))}
                    labelWidth={196}
                    rowHeight={24}
                    highlight={PRIMARY_REGION}
                    fill
                  />
                </div>
              </div>

              {/* Source-portfolio contribution. */}
              <div
                style={{
                  width: 300,
                  flexShrink: 0,
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <Eyebrow>Contribution by source portfolio</Eyebrow>
                <div
                  style={{
                    marginTop: 13,
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    gap: 11,
                  }}
                >
                  {[a, b].map((c, i) => (
                    <div
                      key={c.id}
                      style={{
                        padding: "13px 15px",
                        borderRadius: 7,
                        border: `1px solid ${i === 0 ? C.navy500 : C.line}`,
                        background:
                          i === 0 ? "rgba(145,157,209,0.07)" : "rgba(255,255,255,0.02)",
                        opacity: fade(frame, s(2.0) + i * s(0.2), s(0.5)),
                      }}
                    >
                      <div
                        style={{
                          fontSize: TYPE.micro,
                          color: C.ink500,
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {c.label}
                      </div>
                      <div
                        style={{
                          marginTop: 5,
                          fontSize: TYPE.heading,
                          fontWeight: 600,
                          color: C.ink100,
                          letterSpacing: "-0.02em",
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        {signedMoney(c.delta)}
                      </div>
                      <div style={{ marginTop: 4, fontSize: TYPE.micro, color: C.ink500 }}>
                        {money(c.prior)} → {money(c.current)}
                      </div>
                    </div>
                  ))}
                  <div
                    style={{
                      marginTop: 4,
                      padding: "11px 15px",
                      borderRadius: 7,
                      border: `1px solid ${C.line}`,
                      background: "rgba(255,255,255,0.018)",
                      opacity: fade(frame, s(2.6), s(0.5)),
                    }}
                  >
                    <div style={{ fontSize: TYPE.micro, color: C.ink500 }}>
                      Consolidated movement
                    </div>
                    <div
                      style={{
                        marginTop: 4,
                        fontSize: TYPE.subheading,
                        fontWeight: 600,
                        color: C.ink100,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {signedMoney(d.funded_balance)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </DashboardPanel>

          <ChatColumn>
            {/* The prior turn, scrolled up — this is one continuous conversation. */}
            <div style={{ opacity: 0.4 * fade(frame, s(T.priorTurn), s(0.4)) }}>
              <UserMessage text={miTurn(0).question} />
            </div>
            <div style={{ opacity: fade(frame, s(T.question), s(0.4)) }}>
              <UserMessage text={TURN.question} />
            </div>
            {frame >= s(T.thinking) && frame < answerStart ? (
              <Thinking phase={(frame - s(T.thinking)) / 4} />
            ) : null}
            {frame >= answerStart ? (
              <AgentMessage text={answerText} showCursor={!answerDone} />
            ) : null}

            {frame >= tablesAt ? (
              <div
                style={{
                  opacity: fade(frame, tablesAt, s(0.6)),
                  background: C.surfaceArtifact,
                  border: `1px solid ${C.surfaceArtifactLine}`,
                  borderRadius: 10,
                  padding: "15px 17px",
                }}
              >
                <Eyebrow color={C.ink400}>Regional contribution to balance change</Eyebrow>
                <div style={{ marginTop: 12 }}>
                  <DataTable
                    columns={[
                      { key: "region", label: "Region" },
                      { key: "prior", label: periodLabel(PRIOR_PERIOD.period), align: "right" },
                      { key: "current", label: periodLabel(CURRENT_PERIOD.period), align: "right" },
                      { key: "delta", label: "Change", align: "right" },
                    ]}
                    rows={contributions.slice(0, 4).map((c) => ({
                      region: c.region,
                      prior: money(c.start),
                      current: money(c.end),
                      delta: signedMoney(c.delta),
                    }))}
                    highlightRow={PRIMARY_REGION}
                    fontSize={TYPE.label}
                  />
                </div>
                <div style={{ marginTop: 11, fontSize: TYPE.micro, color: C.ink500, lineHeight: 1.5 }}>
                  Loans completed in {periodLabel(CURRENT_PERIOD.period)} carry{" "}
                  {money(MOVEMENT.completionsBalanceInPrimaryRegion)} of balance in the{" "}
                  {PRIMARY_REGION}
                  {MOVEMENT.completionsShareOfPrimaryRegion != null
                    ? ` — ${percent(
                        MOVEMENT.completionsShareOfPrimaryRegion * 100,
                        0,
                      )} of that region's increase.`
                    : "."}
                </div>
              </div>
            ) : null}
          </ChatColumn>
        </div>
      </AppWindow>
    </Frame>
  );
};
