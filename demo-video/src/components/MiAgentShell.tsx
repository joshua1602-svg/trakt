/**
 * The Trakt MI Agent surface, reconstructed for the film.
 *
 * This is a faithful reconstruction of `frontend/mi-agent-ui`'s three-region
 * layout — header chrome with the portfolio and reporting-date selectors, the
 * funded dashboard panel, and the agent chat column — rendered from the captured
 * production state in `demo_metrics.json`:
 *
 *   * the portfolio selector options come from `/mi/source-portfolios`;
 *   * the reporting dates come from `/mi/snapshots`;
 *   * the KPI tiles come from `/mi/snapshot`;
 *   * the chat answers and their artifacts are the verbatim `/mi/query` envelopes.
 *
 * It is a reconstruction rather than the live app because the live app fetches
 * asynchronously and animates on mount, neither of which is frame-deterministic.
 * Every value shown is nonetheless the value the production endpoint returned —
 * nothing here is a mock-up figure.
 */

import React from "react";
import { C, FONT, LAYOUT, PANEL_BORDER, THEME, TYPE } from "../design/tokens";
import { Eyebrow, KpiTile, Pill, TraktMark } from "./primitives";
import { CLIENT, DASHBOARD, SUMMARY } from "../data/fixtures";
import { count, longDate, money, percent, shortDate, years } from "../design/format";

// --------------------------------------------------------------------------- //
// Header
// --------------------------------------------------------------------------- //
const Selector: React.FC<{
  label: string;
  value: string;
  open?: boolean;
  options?: { label: string; selected?: boolean }[];
  width?: number;
}> = ({ label, value, open = false, options = [], width = 250 }) => (
  <div style={{ position: "relative", width }}>
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "9px 13px",
        border: `1px solid ${open ? C.navy500 : C.line}`,
        borderRadius: LAYOUT.radiusSmall,
        background: open ? "rgba(145,157,209,0.07)" : "rgba(17,22,46,0.6)",
      }}
    >
      <span
        style={{
          fontSize: 10,
          letterSpacing: "0.13em",
          textTransform: "uppercase",
          color: C.ink500,
          fontWeight: 600,
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: TYPE.label,
          color: C.ink100,
          fontWeight: 500,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          flex: 1,
        }}
      >
        {value}
      </span>
      <svg width={11} height={11} viewBox="0 0 16 16" style={{ flexShrink: 0 }}>
        <path
          d="M3.5 6 L8 10.5 L12.5 6"
          fill="none"
          stroke={C.ink400}
          strokeWidth={1.8}
          strokeLinecap="round"
        />
      </svg>
    </div>
    {open && options.length ? (
      <div
        style={{
          position: "absolute",
          top: "calc(100% + 6px)",
          left: 0,
          right: 0,
          background: C.navy850,
          border: `1px solid ${C.navy500}`,
          borderRadius: LAYOUT.radiusSmall,
          boxShadow: "0 18px 44px rgba(0,0,0,0.55)",
          overflow: "hidden",
          zIndex: 10,
        }}
      >
        {options.map((o) => (
          <div
            key={o.label}
            style={{
              padding: "10px 13px",
              fontSize: TYPE.label,
              color: o.selected ? C.ink100 : C.ink300,
              fontWeight: o.selected ? 600 : 400,
              background: o.selected ? "rgba(145,157,209,0.10)" : undefined,
              borderLeft: `2px solid ${o.selected ? C.peri400 : "transparent"}`,
              whiteSpace: "nowrap",
            }}
          >
            {o.label}
          </div>
        ))}
      </div>
    ) : null}
  </div>
);

export const MiAgentHeader: React.FC<{
  portfolioValue: string;
  selectorOpen?: boolean;
  reportingDate: string;
}> = ({ portfolioValue, selectorOpen = false, reportingDate }) => {
  const lenses = DASHBOARD.sourcePortfolios?.lenses ?? [];
  const options = [
    { label: "All Portfolios", selected: portfolioValue === "All Portfolios" },
    ...lenses
      .filter((l) => l.kind === "cohort")
      .map((l) => ({ label: l.label, selected: portfolioValue === l.label })),
  ];
  return (
    <div
      style={{
        height: 58,
        display: "flex",
        alignItems: "center",
        gap: 18,
        padding: "0 20px",
        borderBottom: `1px solid ${C.line}`,
        background: "rgba(17,22,46,0.72)",
        flexShrink: 0,
        position: "relative",
        zIndex: 5,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <TraktMark size={31} />
        <div style={{ lineHeight: 1.15 }}>
          <span style={{ fontSize: TYPE.label, fontWeight: 600, color: C.ink100 }}>
            Trakt
          </span>
          <span style={{ fontSize: TYPE.label, color: C.ink400 }}> · MI Agent</span>
        </div>
      </div>
      <div style={{ width: 1, height: 28, background: C.line }} />
      <Selector
        label="Portfolio"
        value={portfolioValue}
        open={selectorOpen}
        options={options}
        width={272}
      />
      <Selector
        label="Reporting date"
        value={shortDate(reportingDate)}
        width={268}
      />
      <div style={{ flex: 1 }} />
      <Pill tone="accent">{CLIENT.displayName}</Pill>
    </div>
  );
};

// --------------------------------------------------------------------------- //
// Dashboard panel
// --------------------------------------------------------------------------- //
/**
 * The dashboard's headline tiles.
 *
 * These come from the GOVERNED SUMMARY (`movement_summary.portfolio_summary`) —
 * the same service that produced the MI Agent's answer — rather than from
 * `/mi/snapshot`. Both are real product endpoints, but they carry two different
 * borrower-age definitions: `/mi/snapshot` reports a balance-WEIGHTED average age
 * (72.5 years on this book) while the summary and movement services report the
 * simple average (71.4). Showing both on one screen would put two figures for
 * "borrower age" side by side and read as an inconsistency, so the film uses one
 * definition throughout — the one the answers quote.
 */
export const FundedKpiRow: React.FC<{ visibleCount?: number }> = ({
  visibleCount = 6,
}) => {
  const m = SUMMARY.metrics;
  const snapshotById = new Map(DASHBOARD.fundedSnapshot.kpis.map((k) => [k.id, k]));
  const avgBalance = snapshotById.get("avg_balance");
  const originalLtv = snapshotById.get("wa_original_ltv");
  const tiles = [
    { label: "Current funded balance", value: money(m.funded_balance) },
    { label: "Loans funded", value: count(m.loan_count) },
    { label: "Weighted avg current LTV", value: percent(m.wa_ltv_points) },
    { label: "Weighted avg interest rate", value: percent(m.wa_interest_rate, 2) },
    { label: "Average youngest-borrower age", value: years(m.avg_borrower_age) },
    {
      label: avgBalance?.label ?? "Average loan balance",
      value: avgBalance?.value ?? "—",
    },
    {
      label: originalLtv?.label ?? "Weighted avg original LTV",
      value: originalLtv?.value ?? "—",
    },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
      {tiles.slice(0, visibleCount).map((k, i) => (
        <KpiTile key={k.label} label={k.label} value={k.value} emphasis={i === 0} />
      ))}
    </div>
  );
};

export const DashboardPanel: React.FC<{
  children?: React.ReactNode;
  reportingDate: string;
  title?: string;
}> = ({ children, reportingDate, title = "Funded portfolio" }) => (
  <div
    style={{
      flex: 1,
      minWidth: 0,
      background: C.surfaceDashboard,
      borderRight: `1px solid ${C.surfaceDashboardLine}`,
      padding: 22,
      display: "flex",
      flexDirection: "column",
      gap: 16,
      overflow: "hidden",
    }}
  >
    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
      <div>
        <Eyebrow>{title}</Eyebrow>
        <div style={{ marginTop: 5, fontSize: TYPE.body, fontWeight: 600, color: C.ink100 }}>
          {CLIENT.displayName}
        </div>
      </div>
      <div style={{ textAlign: "right" }}>
        <Eyebrow>Data cut-off</Eyebrow>
        <div style={{ marginTop: 5, fontSize: TYPE.bodySmall, color: C.ink300 }}>
          {longDate(reportingDate)}
        </div>
      </div>
    </div>
    {children}
  </div>
);

// --------------------------------------------------------------------------- //
// Chat column
// --------------------------------------------------------------------------- //
export const ChatColumn: React.FC<{ children: React.ReactNode; width?: number }> = ({
  children,
  width = 660,
}) => (
  <div
    style={{
      width,
      flexShrink: 0,
      background: C.surfaceChat,
      padding: 22,
      display: "flex",
      flexDirection: "column",
      gap: 14,
      overflow: "hidden",
    }}
  >
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
      <Eyebrow color={C.peri300}>MI Agent</Eyebrow>
      <Pill tone="positive">
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: 999,
            background: THEME.positive,
            display: "inline-block",
          }}
        />
        Governed dataset
      </Pill>
    </div>
    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 14, minHeight: 0 }}>
      {children}
    </div>
  </div>
);

export const UserMessage: React.FC<{ text: string; opacity?: number }> = ({
  text,
  opacity = 1,
}) => (
  <div style={{ display: "flex", justifyContent: "flex-end", opacity }}>
    <div
      style={{
        maxWidth: "84%",
        padding: "12px 16px",
        borderRadius: `${LAYOUT.radius}px ${LAYOUT.radius}px 4px ${LAYOUT.radius}px`,
        background: C.navy700,
        border: `1px solid ${C.navy500}`,
        fontSize: TYPE.body,
        lineHeight: 1.45,
        color: C.ink100,
      }}
    >
      {text}
    </div>
  </div>
);

export const AgentMessage: React.FC<{
  text: string;
  opacity?: number;
  showCursor?: boolean;
}> = ({ text, opacity = 1, showCursor = false }) => (
  <div style={{ display: "flex", gap: 12, opacity }}>
    <TraktMark size={28} />
    <div
      style={{
        flex: 1,
        minWidth: 0,
        padding: "13px 17px",
        borderRadius: `${LAYOUT.radius}px ${LAYOUT.radius}px ${LAYOUT.radius}px 4px`,
        background: "rgba(255,255,255,0.028)",
        border: PANEL_BORDER,
        fontSize: TYPE.body,
        lineHeight: 1.52,
        color: C.ink100,
      }}
    >
      {text}
      {showCursor ? (
        <span
          style={{
            display: "inline-block",
            width: 2,
            height: TYPE.body,
            background: C.peri300,
            marginLeft: 2,
            verticalAlign: "text-bottom",
          }}
        />
      ) : null}
    </div>
  </div>
);

/** The three-dot "thinking" indicator, shown for well under a second. */
export const Thinking: React.FC<{ opacity?: number; phase: number }> = ({
  opacity = 1,
  phase,
}) => (
  <div style={{ display: "flex", gap: 12, opacity, alignItems: "center" }}>
    <TraktMark size={28} />
    <div
      style={{
        padding: "15px 19px",
        borderRadius: LAYOUT.radius,
        background: "rgba(255,255,255,0.028)",
        border: PANEL_BORDER,
        display: "flex",
        gap: 6,
      }}
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          style={{
            width: 6,
            height: 6,
            borderRadius: 999,
            background: C.peri300,
            opacity: 0.28 + 0.72 * Math.max(0, Math.cos((phase - i * 0.6) * 1.1)),
          }}
        />
      ))}
    </div>
  </div>
);

/** A supporting-evidence panel beneath an answer, as the product renders one. */
export const EvidencePanel: React.FC<{
  title: string;
  children: React.ReactNode;
  opacity?: number;
  footnote?: string;
}> = ({ title, children, opacity = 1, footnote }) => (
  <div
    style={{
      opacity,
      background: C.surfaceArtifact,
      border: `1px solid ${C.surfaceArtifactLine}`,
      borderRadius: LAYOUT.radius,
      padding: "16px 18px",
      display: "flex",
      flexDirection: "column",
      gap: 13,
      minWidth: 0,
    }}
  >
    <Eyebrow color={C.ink400}>{title}</Eyebrow>
    {children}
    {footnote ? (
      <div style={{ fontSize: TYPE.micro, color: C.ink500, lineHeight: 1.45 }}>
        {footnote}
      </div>
    ) : null}
  </div>
);

/** The full application window, with the browser-chrome-free device frame. */
export const AppWindow: React.FC<{
  children: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ children, style }) => (
  <div
    style={{
      flex: 1,
      minHeight: 0,
      borderRadius: LAYOUT.radius,
      border: `1px solid ${C.navy600}`,
      background: C.navy950,
      overflow: "hidden",
      display: "flex",
      flexDirection: "column",
      boxShadow: "0 34px 90px rgba(0,0,0,0.5)",
      fontFamily: FONT.sans,
      ...style,
    }}
  >
    {children}
  </div>
);
