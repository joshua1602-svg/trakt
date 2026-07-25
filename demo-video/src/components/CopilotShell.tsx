/**
 * A controlled Microsoft 365 Copilot-style window.
 *
 * What this is: a neutral representation of the Copilot chat surface, showing
 * Trakt as a named agent inside it, with every answer visibly attributed to a
 * Trakt action. The actions shown are the three that actually exist in
 * `deploy/copilot-agent/ai-plugin.json`:
 *
 *     askTraktMi · getLatestInvestorDeck · getLatestCanonicalTape
 *
 * What this is NOT: a reproduction of Microsoft's product. No Microsoft logo,
 * icon, font or other brand asset is embedded — the chrome is a neutral light
 * surface with the product's plain-text name, and Trakt is not implied to own or
 * supply Copilot. The agent card shows the Trakt mark, because the agent is ours.
 *
 * The figures come from the captured `/v1/copilot/mi/query` responses, which
 * delegate to the same `/mi/query` handler the MI Agent uses — so the numbers here
 * are the same governed values by construction, not a re-statement.
 */

import React from "react";
import { C, FONT, LAYOUT, THEME, TYPE } from "../design/tokens";
import { TraktMark } from "./primitives";

/** Light Copilot-surface palette. Neutral greys — deliberately not a brand ramp. */
export const COPILOT = {
  page: "#f6f6f8",
  panel: "#ffffff",
  panelAlt: "#fbfbfd",
  border: "#e2e2e8",
  borderStrong: "#cfcfd8",
  ink: "#1b1b1f",
  inkSoft: "#4a4a55",
  inkFaint: "#77777f",
  userBubble: "#eef0f6",
} as const;

export const CopilotWindow: React.FC<{
  children: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ children, style }) => (
  <div
    style={{
      flex: 1,
      minHeight: 0,
      borderRadius: LAYOUT.radius,
      border: `1px solid ${COPILOT.borderStrong}`,
      background: COPILOT.page,
      overflow: "hidden",
      display: "flex",
      flexDirection: "column",
      boxShadow: "0 34px 90px rgba(0,0,0,0.5)",
      fontFamily: FONT.sans,
      color: COPILOT.ink,
      ...style,
    }}
  >
    {children}
  </div>
);

export const CopilotHeader: React.FC<{ agentName: string }> = ({ agentName }) => (
  <div
    style={{
      height: 54,
      display: "flex",
      alignItems: "center",
      gap: 14,
      padding: "0 20px",
      background: COPILOT.panel,
      borderBottom: `1px solid ${COPILOT.border}`,
      flexShrink: 0,
    }}
  >
    <span style={{ fontSize: TYPE.label, fontWeight: 600, color: COPILOT.ink }}>
      Microsoft 365 Copilot
    </span>
    <div style={{ width: 1, height: 22, background: COPILOT.border }} />
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 9,
        padding: "5px 12px 5px 6px",
        borderRadius: 999,
        background: COPILOT.panelAlt,
        border: `1px solid ${COPILOT.border}`,
      }}
    >
      <TraktMark size={22} />
      <span style={{ fontSize: TYPE.label, fontWeight: 600, color: COPILOT.ink }}>
        {agentName}
      </span>
      <span style={{ fontSize: TYPE.micro, color: COPILOT.inkFaint }}>agent</span>
    </div>
    <div style={{ flex: 1 }} />
    <span style={{ fontSize: TYPE.micro, color: COPILOT.inkFaint }}>
      Grounded in Trakt · answers are not model estimates
    </span>
  </div>
);

export const CopilotUserMessage: React.FC<{ text: string; opacity?: number }> = ({
  text,
  opacity = 1,
}) => (
  <div style={{ display: "flex", justifyContent: "flex-end", opacity }}>
    <div
      style={{
        maxWidth: "78%",
        padding: "12px 17px",
        borderRadius: "16px 16px 5px 16px",
        background: COPILOT.userBubble,
        fontSize: TYPE.body,
        lineHeight: 1.45,
        color: COPILOT.ink,
      }}
    >
      {text}
    </div>
  </div>
);

/**
 * The action-call chip. This is the important honesty signal in the scene: the
 * answer is preceded by a visible statement that Copilot invoked a named Trakt
 * action, so the film never implies Copilot computed anything itself.
 */
export const ActionCallChip: React.FC<{
  action: string;
  detail?: string;
  opacity?: number;
  running?: boolean;
}> = ({ action, detail, opacity = 1, running = false }) => (
  <div
    style={{
      opacity,
      display: "inline-flex",
      alignItems: "center",
      gap: 10,
      padding: "7px 13px",
      borderRadius: 8,
      background: COPILOT.panelAlt,
      border: `1px solid ${COPILOT.border}`,
      alignSelf: "flex-start",
    }}
  >
    <TraktMark size={19} />
    <span style={{ fontSize: TYPE.micro, color: COPILOT.inkSoft }}>
      {running ? "Calling" : "Called"} Trakt action
    </span>
    <span
      style={{
        fontFamily: FONT.mono,
        fontSize: TYPE.micro,
        fontWeight: 600,
        color: COPILOT.ink,
      }}
    >
      {action}
    </span>
    {detail ? (
      <span style={{ fontSize: TYPE.micro, color: COPILOT.inkFaint }}>{detail}</span>
    ) : null}
    {!running ? (
      <svg width={13} height={13} viewBox="0 0 16 16">
        <path
          d="M3 8.5 L6.4 12 L13 4.5"
          fill="none"
          stroke={THEME.positive}
          strokeWidth={2.2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ) : null}
  </div>
);

export const CopilotAgentMessage: React.FC<{
  text: string;
  opacity?: number;
  showCursor?: boolean;
  attribution?: string;
}> = ({ text, opacity = 1, showCursor = false, attribution }) => (
  <div style={{ opacity, display: "flex", flexDirection: "column", gap: 9 }}>
    <div
      style={{
        padding: "15px 19px",
        borderRadius: 16,
        background: COPILOT.panel,
        border: `1px solid ${COPILOT.border}`,
        fontSize: TYPE.body,
        lineHeight: 1.55,
        color: COPILOT.ink,
      }}
    >
      {text}
      {showCursor ? (
        <span
          style={{
            display: "inline-block",
            width: 2,
            height: TYPE.body,
            background: COPILOT.inkSoft,
            marginLeft: 2,
            verticalAlign: "text-bottom",
          }}
        />
      ) : null}
    </div>
    {attribution ? (
      <div
        style={{
          fontSize: TYPE.micro,
          color: COPILOT.inkFaint,
          display: "flex",
          alignItems: "center",
          gap: 7,
        }}
      >
        <TraktMark size={16} />
        {attribution}
      </div>
    ) : null}
  </div>
);

/** A secure-download result card for the investor-deck action. */
export const CopilotArtefactCard: React.FC<{
  fileName: string;
  reportingPeriod: string;
  expiresInSeconds: number;
  opacity?: number;
}> = ({ fileName, reportingPeriod, expiresInSeconds, opacity = 1 }) => (
  <div
    style={{
      opacity,
      padding: "16px 19px",
      borderRadius: 16,
      background: COPILOT.panel,
      border: `1px solid ${COPILOT.border}`,
      display: "flex",
      alignItems: "center",
      gap: 16,
    }}
  >
    <div
      style={{
        width: 44,
        height: 44,
        borderRadius: 9,
        background: "#eceef5",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
      }}
    >
      <svg width={22} height={22} viewBox="0 0 24 24">
        <path
          d="M14 3 H7 a2 2 0 0 0 -2 2 v14 a2 2 0 0 0 2 2 h10 a2 2 0 0 0 2 -2 V8 Z"
          fill="none"
          stroke={C.navy700}
          strokeWidth={1.6}
          strokeLinejoin="round"
        />
        <path d="M14 3 V8 H19" fill="none" stroke={C.navy700} strokeWidth={1.6} />
      </svg>
    </div>
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: TYPE.body, fontWeight: 600, color: COPILOT.ink }}>
        {fileName}
      </div>
      <div style={{ marginTop: 3, fontSize: TYPE.bodySmall, color: COPILOT.inkSoft }}>
        Reporting period {reportingPeriod} · latest approved artefact
      </div>
      <div style={{ marginTop: 6, fontSize: TYPE.micro, color: COPILOT.inkFaint }}>
        Secure download link, expires in {Math.round(expiresInSeconds / 60)} minutes
      </div>
    </div>
    <div
      style={{
        padding: "9px 16px",
        borderRadius: 7,
        background: C.navy700,
        color: "#ffffff",
        fontSize: TYPE.label,
        fontWeight: 600,
        flexShrink: 0,
      }}
    >
      Download
    </div>
  </div>
);

/** The prompt bar at the foot of the Copilot window. */
export const CopilotPromptBar: React.FC<{ placeholder?: string }> = ({
  placeholder = "Ask Trakt about the portfolio…",
}) => (
  <div
    style={{
      flexShrink: 0,
      padding: "14px 20px 18px",
      background: COPILOT.page,
      borderTop: `1px solid ${COPILOT.border}`,
    }}
  >
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "12px 16px",
        borderRadius: 12,
        background: COPILOT.panel,
        border: `1px solid ${COPILOT.borderStrong}`,
      }}
    >
      <TraktMark size={21} />
      <span style={{ fontSize: TYPE.bodySmall, color: COPILOT.inkFaint, flex: 1 }}>
        {placeholder}
      </span>
      <span style={{ fontSize: TYPE.micro, color: COPILOT.inkFaint }}>
        Trakt agent selected
      </span>
    </div>
  </div>
);
