/**
 * Scene 4 — Consolidated MI Agent.
 *
 * The reconstructed Trakt MI Agent, showing the consolidated view, the portfolio
 * selector opening to reveal All Portfolios / both books, the funded KPI tiles from
 * `/mi/snapshot`, and the first conversation turn: "Summarise the portfolio."
 *
 * The answer text is the verbatim `/mi/query` answer. The supporting evidence
 * panel is the KPI artifact and the regional exposure chart from the SAME envelope.
 * No internal query spec, trace or engine metadata appears — the capture step
 * removed those from the fixture, so they cannot reach a frame.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { Frame } from "../components/Frame";
import { BarList, Eyebrow, Sparkline } from "../components/primitives";
import {
  AgentMessage,
  AppWindow,
  ChatColumn,
  DashboardPanel,
  FundedKpiRow,
  MiAgentHeader,
  Thinking,
  UserMessage,
} from "../components/MiAgentShell";
import { C, TYPE } from "../design/tokens";
import { enter, fade, s, typed, typedDuration } from "../design/motion";
import {
  CURRENT_PERIOD,
  SERIES,
  SUMMARY,
  miTurn,
  PRIMARY_REGION,
} from "../data/fixtures";
import { money, noBreakSigns, percent, periodLabel, share } from "../design/format";
import type { Caption } from "../timeline";

const TURN = miTurn(0);

/** Scene beats, in seconds. Kept here so the scene reads top-to-bottom. */
const T = {
  windowIn: 0.2,
  kpis: 1.4,
  selectorOpen: 4.0,
  selectorClose: 6.6,
  question: 7.6,
  thinking: 9.6,
  answer: 10.5,
} as const;

export const Scene4MiSummary: React.FC<{ captions: Caption[] }> = ({ captions }) => {
  const frame = useCurrentFrame();
  const answerStart = s(T.answer);
  const answerText = noBreakSigns(typed(TURN.answer, frame, answerStart, 3.4));
  const answerDone = frame > answerStart + typedDuration(TURN.answer, 3.4);
  const evidenceAt = answerStart + typedDuration(TURN.answer, 3.4) + s(0.2);
  const selectorOpen = frame >= s(T.selectorOpen) && frame < s(T.selectorClose);

  const regions = SUMMARY.topRegions.slice(0, 7);

  return (
    <Frame captions={captions} chapter="Trakt MI Agent">
      <AppWindow style={{ ...enter(frame, s(T.windowIn), s(0.8), 14) }}>
        <MiAgentHeader
          portfolioValue="All Portfolios"
          selectorOpen={selectorOpen}
          reportingDate={CURRENT_PERIOD.reportingDate}
        />
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <DashboardPanel reportingDate={CURRENT_PERIOD.reportingDate}>
            <div style={{ opacity: fade(frame, s(T.kpis), s(0.7)) }}>
              <FundedKpiRow />
            </div>

            <div
              style={{
                display: "flex",
                gap: 16,
                flex: 1,
                minHeight: 0,
                opacity: fade(frame, s(T.kpis + 0.7), s(0.7)),
              }}
            >
              {/* Regional exposure — the largest concentrations. */}
              <div style={{ flex: 1.25, minWidth: 0, display: "flex", flexDirection: "column" }}>
                <Eyebrow>Funded balance by region</Eyebrow>
                <div style={{ marginTop: 14, flex: 1 }}>
                  <BarList
                    data={regions.map((r) => ({ label: r.region, value: r.balance }))}
                    formatValue={money}
                    progress={fade(frame, s(T.kpis + 1.0), s(1.1))}
                    labelWidth={196}
                    rowHeight={24}
                    highlight={PRIMARY_REGION}
                    fill
                  />
                </div>
              </div>

              {/* Portfolio split + the three-period series. */}
              <div
                style={{
                  width: 268,
                  flexShrink: 0,
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  gap: 20,
                }}
              >
                <div>
                  <Eyebrow>Portfolio split</Eyebrow>
                  <div style={{ marginTop: 13, display: "flex", flexDirection: "column", gap: 10 }}>
                    {SUMMARY.cohorts.map((c, i) => {
                      const balance = SUMMARY.cohortBalances[c.id];
                      const pct = balance / (SUMMARY.metrics.funded_balance ?? 1);
                      return (
                        <div key={c.id}>
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              gap: 10,
                              fontSize: TYPE.label,
                            }}
                          >
                            <span
                              style={{
                                color: C.ink300,
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                              }}
                            >
                              {c.label}
                            </span>
                            <span
                              style={{
                                color: C.ink100,
                                fontWeight: 600,
                                fontVariantNumeric: "tabular-nums",
                                flexShrink: 0,
                              }}
                            >
                              {money(balance)}
                            </span>
                          </div>
                          <div
                            style={{
                              marginTop: 6,
                              height: 5,
                              borderRadius: 3,
                              background: "rgba(255,255,255,0.05)",
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${pct * 100 * fade(frame, s(T.kpis + 1.2), s(1.0))}%`,
                                height: "100%",
                                background: i === 0 ? C.peri400 : C.navy500,
                              }}
                            />
                          </div>
                          <div style={{ marginTop: 4, fontSize: TYPE.micro, color: C.ink500 }}>
                            {share(pct, 1)} of the book
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div>
                  <Eyebrow>Funded balance trend</Eyebrow>
                  <div style={{ marginTop: 16 }}>
                    <Sparkline
                      points={SERIES.map((p) => ({
                        label: periodLabel(p.period),
                        value: p.fundedBalance ?? 0,
                      }))}
                      width={244}
                      height={82}
                      progress={fade(frame, s(T.kpis + 1.4), s(1.2))}
                    />
                  </div>
                </div>
              </div>
            </div>
          </DashboardPanel>

          <ChatColumn>
            <div style={{ opacity: fade(frame, s(T.question), s(0.4)) }}>
              <UserMessage text={TURN.question} />
            </div>
            {frame >= s(T.thinking) && frame < answerStart ? (
              <Thinking phase={(frame - s(T.thinking)) / 4} />
            ) : null}
            {frame >= answerStart ? (
              <AgentMessage text={answerText} showCursor={!answerDone} />
            ) : null}

            {/* Supporting evidence: the KPI artifact from the same envelope. */}
            {frame >= evidenceAt ? (
              <div
                style={{
                  opacity: fade(frame, evidenceAt, s(0.6)),
                  display: "flex",
                  flexDirection: "column",
                  gap: 11,
                }}
              >
                <Eyebrow color={C.ink500}>Supporting evidence</Eyebrow>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 9 }}>
                  {[
                    { label: "Loans funded", value: SUMMARY.metrics.loan_count },
                    { label: "Funded balance", value: SUMMARY.metrics.funded_balance },
                    { label: "WA current LTV", value: SUMMARY.metrics.wa_ltv_points },
                    { label: "Avg borrower age", value: SUMMARY.metrics.avg_borrower_age },
                  ].map((tile, i) => (
                    <div
                      key={tile.label}
                      style={{
                        padding: "11px 13px",
                        borderRadius: 6,
                        background: C.surfaceArtifact,
                        border: `1px solid ${C.surfaceArtifactLine}`,
                        opacity: fade(frame, evidenceAt + i * s(0.1), s(0.4)),
                      }}
                    >
                      <div style={{ fontSize: TYPE.micro, color: C.ink500 }}>{tile.label}</div>
                      <div
                        style={{
                          marginTop: 4,
                          fontSize: TYPE.body,
                          fontWeight: 600,
                          color: C.ink100,
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        {i === 0
                          ? new Intl.NumberFormat("en-GB").format(tile.value ?? 0)
                          : i === 1
                            ? money(tile.value)
                            : i === 2
                              ? percent(tile.value)
                              : `${(tile.value ?? 0).toFixed(1)} years`}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </ChatColumn>
        </div>
      </AppWindow>
    </Frame>
  );
};
