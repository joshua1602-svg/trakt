/**
 * Scene 6 — Microsoft 365 Copilot.
 *
 * Trakt as an agent inside Copilot, answering the same two questions with the same
 * governed values, then retrieving the latest approved investor deck.
 *
 * Three things this scene is careful about:
 *
 *   1. every answer is preceded by a visible "called Trakt action askTraktMi"
 *      chip, so Copilot is never shown computing a portfolio figure itself;
 *   2. the answers are the captured `/v1/copilot/mi/query` responses, which
 *      delegate to the same `/mi/query` handler the MI Agent uses — the assertion
 *      gate compares the rendered figures across both surfaces;
 *   3. the deck action's download link is a redacted placeholder in the fixture.
 *      A real signed URL is a bearer credential and the safety scan fails the
 *      render if one is present.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { Frame } from "../components/Frame";
import { Eyebrow, Pill } from "../components/primitives";
import {
  ActionCallChip,
  COPILOT as CP,
  CopilotAgentMessage,
  CopilotArtefactCard,
  CopilotHeader,
  CopilotPromptBar,
  CopilotUserMessage,
  CopilotWindow,
} from "../components/CopilotShell";
import { C, LAYOUT, TYPE } from "../design/tokens";
import { enter, fade, s, typed, typedDuration } from "../design/motion";
import { COPILOT, CURRENT_PERIOD, MOVEMENT, contribution, miTurn } from "../data/fixtures";
import { longDate, money, noBreakSigns, percent, signedMoney } from "../design/format";
import type { Caption } from "../timeline";

const Q1 = miTurn(0).question;
const Q2 = miTurn(1).question;
const A1 = COPILOT.answers[0]?.answer ?? "";
const A2 = COPILOT.answers[1]?.answer ?? "";
const DECK = COPILOT.artefactAction;

const CHARS = 4.2;

const T = {
  windowIn: 0.2,
  q1: 1.2,
  action1: 2.4,
  a1: 3.2,
  q2: 13.0,
  action2: 14.0,
  a2: 14.8,
  q3: 24.0,
  action3: 25.0,
  a3: 25.9,
} as const;

/** A compact side rail proving the two surfaces agree on the same numbers. */
const AgreementRail: React.FC<{ frame: number }> = ({ frame }) => {
  const a = contribution("A");
  const b = contribution("B");
  const rows = [
    { label: "Funded balance", value: money(MOVEMENT.current.funded_balance) },
    { label: "Movement", value: signedMoney(MOVEMENT.delta.funded_balance) },
    { label: "WA current LTV", value: percent(MOVEMENT.current.wa_ltv_points) },
    {
      label: "Avg borrower age",
      value: `${(MOVEMENT.current.avg_borrower_age ?? 0).toFixed(1)} years`,
    },
    { label: a.label, value: signedMoney(a.delta) },
    { label: b.label, value: signedMoney(b.delta) },
  ];
  return (
    <div
      style={{
        width: 322,
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        gap: 13,
        ...enter(frame, s(T.a1 + 1.4), s(0.8)),
      }}
    >
      <div>
        <Eyebrow color={C.peri300}>One governed answer</Eyebrow>
        <div style={{ marginTop: 7, fontSize: TYPE.bodySmall, color: C.ink400, lineHeight: 1.5 }}>
          Copilot and the Trakt MI Agent return the same values because both call the
          same deterministic MI engine.
        </div>
      </div>
      <div
        style={{
          border: `1px solid ${C.line}`,
          borderRadius: LAYOUT.radiusSmall,
          overflow: "hidden",
        }}
      >
        {rows.map((r, i) => (
          <div
            key={r.label}
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: 12,
              padding: "11px 14px",
              background: i % 2 === 0 ? "rgba(255,255,255,0.022)" : "transparent",
              opacity: fade(frame, s(T.a1 + 1.7) + i * s(0.11), s(0.4)),
            }}
          >
            <span
              style={{
                fontSize: TYPE.label,
                color: C.ink400,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {r.label}
            </span>
            <span
              style={{
                fontSize: TYPE.label,
                fontWeight: 600,
                color: C.ink100,
                fontVariantNumeric: "tabular-nums",
                flexShrink: 0,
              }}
            >
              {r.value}
            </span>
          </div>
        ))}
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          opacity: fade(frame, s(T.q3 - 1.0), s(0.7)),
        }}
      >
        <Eyebrow style={{ width: "100%", marginBottom: 3 }}>Available Trakt actions</Eyebrow>
        {COPILOT.actions.map((action) => (
          <Pill key={action} tone="accent" style={{ fontFamily: "inherit" }}>
            {action}
          </Pill>
        ))}
      </div>
    </div>
  );
};

export const Scene6Copilot: React.FC<{ captions: Caption[] }> = ({ captions }) => {
  const frame = useCurrentFrame();

  const a1Start = s(T.a1);
  const a2Start = s(T.a2);
  const a3Start = s(T.a3);
  const a1 = noBreakSigns(typed(A1, frame, a1Start, CHARS));
  const a2 = noBreakSigns(typed(A2, frame, a2Start, CHARS));
  const a1Done = frame > a1Start + typedDuration(A1, CHARS);
  const a2Done = frame > a2Start + typedDuration(A2, CHARS);

  // The conversation scrolls as it grows, so the newest turn always sits in view.
  const scroll =
    frame < s(T.q2)
      ? 0
      : frame < s(T.q3)
        ? -Math.min(140, Math.max(0, (frame - s(T.q2)) * 3.4))
        : -Math.min(330, 140 + Math.max(0, (frame - s(T.q3)) * 3.8));

  return (
    <Frame captions={captions} chapter="Microsoft 365 Copilot">
      <div style={{ display: "flex", gap: 30, flex: 1, minHeight: 0 }}>
        <CopilotWindow style={{ ...enter(frame, s(T.windowIn), s(0.8), 14) }}>
          <CopilotHeader agentName={COPILOT.agentName} />
          <div
            style={{
              flex: 1,
              minHeight: 0,
              overflow: "hidden",
              padding: "20px 26px",
              display: "flex",
              flexDirection: "column",
              // A scrolled conversation must dissolve at the top edge rather than
              // being sliced, so a partially-scrolled turn never looks broken.
              maskImage:
                "linear-gradient(to bottom, transparent 0px, black 26px, black 100%)",
              WebkitMaskImage:
                "linear-gradient(to bottom, transparent 0px, black 26px, black 100%)",
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 14,
                transform: `translateY(${scroll}px)`,
              }}
            >
              {/* Turn 1 — summarise the portfolio. */}
              <div style={{ opacity: fade(frame, s(T.q1), s(0.4)) }}>
                <CopilotUserMessage text={Q1} />
              </div>
              {frame >= s(T.action1) ? (
                <ActionCallChip
                  action="askTraktMi"
                  detail={`as at ${longDate(CURRENT_PERIOD.reportingDate)}`}
                  opacity={fade(frame, s(T.action1), s(0.4))}
                  running={frame < a1Start}
                />
              ) : null}
              {frame >= a1Start ? (
                <CopilotAgentMessage
                  text={a1}
                  showCursor={!a1Done}
                  attribution={
                    a1Done ? "Grounded in Trakt · deterministic MI engine" : undefined
                  }
                />
              ) : null}

              {/* Turn 2 — the month-on-month comparison. */}
              {frame >= s(T.q2) ? (
                <div style={{ opacity: fade(frame, s(T.q2), s(0.4)) }}>
                  <CopilotUserMessage text={Q2} />
                </div>
              ) : null}
              {frame >= s(T.action2) ? (
                <ActionCallChip
                  action="askTraktMi"
                  opacity={fade(frame, s(T.action2), s(0.4))}
                  running={frame < a2Start}
                />
              ) : null}
              {frame >= a2Start ? (
                <CopilotAgentMessage
                  text={a2}
                  showCursor={!a2Done}
                  attribution={
                    a2Done ? "Grounded in Trakt · deterministic MI engine" : undefined
                  }
                />
              ) : null}

              {/* Turn 3 — retrieve the latest approved artefact. */}
              {frame >= s(T.q3) ? (
                <div style={{ opacity: fade(frame, s(T.q3), s(0.4)) }}>
                  <CopilotUserMessage text={COPILOT.artefactQuestion} />
                </div>
              ) : null}
              {frame >= s(T.action3) ? (
                <ActionCallChip
                  action="getLatestInvestorDeck"
                  detail="latest approved artefact"
                  opacity={fade(frame, s(T.action3), s(0.4))}
                  running={frame < a3Start}
                />
              ) : null}
              {frame >= a3Start && DECK ? (
                <div style={{ opacity: fade(frame, a3Start, s(0.5)) }}>
                  <CopilotArtefactCard
                    fileName={DECK.fileName}
                    reportingPeriod={DECK.reportingPeriod}
                    expiresInSeconds={DECK.expiresInSeconds}
                  />
                  <div
                    style={{
                      marginTop: 9,
                      fontSize: TYPE.micro,
                      color: CP.inkFaint,
                      lineHeight: 1.5,
                    }}
                  >
                    Trakt resolved the current approved deck and returned a short-lived
                    signed link. Copilot never receives a storage path or a credential.
                  </div>
                </div>
              ) : null}
            </div>
          </div>
          <CopilotPromptBar />
        </CopilotWindow>

        <AgreementRail frame={frame} />
      </div>
    </Frame>
  );
};
