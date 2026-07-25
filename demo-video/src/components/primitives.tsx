/**
 * Shared visual primitives, styled from the product's own design tokens.
 *
 * These mirror `frontend/mi-agent-ui/src/components/ui.tsx` in look and spacing —
 * a card is a card, a KPI tile is a KPI tile — reimplemented as plain inline
 * styles because the Remotion bundle does not run the product's Tailwind build.
 * Anything that carries a business value takes it as a prop from the fixtures.
 */

import React from "react";
import { C, FONT, LAYOUT, PANEL_BORDER, THEME, TYPE } from "../design/tokens";

// --------------------------------------------------------------------------- //
// Brand
// --------------------------------------------------------------------------- //
/** The Trakt mark, matching the product header's rounded navy tile. */
export const TraktMark: React.FC<{ size?: number }> = ({ size = 34 }) => (
  <div
    style={{
      width: size,
      height: size,
      borderRadius: size * 0.28,
      background: C.navy700,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0,
    }}
  >
    <span
      style={{
        fontFamily: FONT.mono,
        fontSize: size * 0.46,
        fontWeight: 700,
        color: C.peri300,
        letterSpacing: "-0.02em",
      }}
    >
      T
    </span>
  </div>
);

/** "Trakt · MI Agent" lock-up, as the product header renders it. */
export const TraktWordmark: React.FC<{
  suffix?: string;
  size?: number;
}> = ({ suffix, size = TYPE.bodySmall }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
    <TraktMark size={size * 1.9} />
    <div style={{ lineHeight: 1.15 }}>
      <span
        style={{
          fontSize: size,
          fontWeight: 600,
          letterSpacing: "-0.015em",
          color: C.ink100,
        }}
      >
        Trakt
      </span>
      {suffix ? (
        <span style={{ fontSize: size, fontWeight: 400, color: C.ink400 }}>
          {" · "}
          {suffix}
        </span>
      ) : null}
    </div>
  </div>
);

// --------------------------------------------------------------------------- //
// Layout
// --------------------------------------------------------------------------- //
export const Card: React.FC<{
  children: React.ReactNode;
  style?: React.CSSProperties;
  accent?: string;
  padding?: number;
}> = ({ children, style, accent, padding = 26 }) => (
  <div
    style={{
      background: C.surfaceDashboard,
      border: PANEL_BORDER,
      borderRadius: LAYOUT.radius,
      padding,
      position: "relative",
      overflow: "hidden",
      ...style,
    }}
  >
    {accent ? (
      <div
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: 3,
          background: accent,
        }}
      />
    ) : null}
    {children}
  </div>
);

/** A small uppercase eyebrow label, used above every panel in the product. */
export const Eyebrow: React.FC<{
  children: React.ReactNode;
  color?: string;
  style?: React.CSSProperties;
}> = ({ children, color = C.ink500, style }) => (
  <div
    style={{
      fontSize: TYPE.micro,
      letterSpacing: "0.14em",
      textTransform: "uppercase",
      color,
      fontWeight: 600,
      ...style,
    }}
  >
    {children}
  </div>
);

export const Divider: React.FC<{ style?: React.CSSProperties }> = ({ style }) => (
  <div style={{ height: 1, background: C.line, ...style }} />
);

// --------------------------------------------------------------------------- //
// KPI tile — the dashboard's headline metric card
// --------------------------------------------------------------------------- //
export const KpiTile: React.FC<{
  label: string;
  value: string;
  delta?: string;
  deltaPositive?: boolean;
  emphasis?: boolean;
  style?: React.CSSProperties;
}> = ({ label, value, delta, deltaPositive, emphasis, style }) => (
  <div
    style={{
      background: emphasis ? "rgba(145,157,209,0.07)" : "rgba(255,255,255,0.018)",
      border: `1px solid ${emphasis ? C.navy500 : C.line}`,
      borderRadius: LAYOUT.radiusSmall,
      padding: "16px 18px",
      display: "flex",
      flexDirection: "column",
      gap: 8,
      minWidth: 0,
      ...style,
    }}
  >
    <div
      style={{
        fontSize: TYPE.micro,
        color: C.ink400,
        letterSpacing: "0.05em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
      }}
    >
      {label}
    </div>
    <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
      <span
        style={{
          fontSize: emphasis ? TYPE.heading : TYPE.subheading,
          fontWeight: 600,
          color: C.ink100,
          letterSpacing: "-0.02em",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </span>
      {delta ? (
        <span
          style={{
            fontSize: TYPE.label,
            fontWeight: 600,
            color: deltaPositive ? THEME.positive : THEME.negative,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {delta}
        </span>
      ) : null}
    </div>
  </div>
);

// --------------------------------------------------------------------------- //
// Charts — plain SVG, so a frame is deterministic and text stays crisp
// --------------------------------------------------------------------------- //
export interface BarDatum {
  label: string;
  value: number;
  colour?: string;
}

/**
 * A horizontal bar list — the product's stratification treatment. `progress`
 * (0..1) grows the bars; the printed value is always the final figure, never an
 * animated one, so no frame can show a wrong number.
 */
export const BarList: React.FC<{
  data: BarDatum[];
  formatValue: (value: number) => string;
  progress?: number;
  labelWidth?: number;
  rowHeight?: number;
  gap?: number;
  barColour?: string;
  highlight?: string;
  /** Distribute the rows over the container's full height. */
  fill?: boolean;
}> = ({
  data,
  formatValue,
  progress = 1,
  labelWidth = 210,
  rowHeight = 26,
  gap = 12,
  barColour = C.peri400,
  highlight,
  fill = false,
}) => {
  const max = Math.max(...data.map((d) => Math.abs(d.value)), 1);
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap,
        ...(fill
          ? { height: "100%", justifyContent: "space-between", gap: undefined }
          : {}),
      }}
    >
      {data.map((d) => {
        const isHighlight = highlight != null && d.label === highlight;
        const width = (Math.abs(d.value) / max) * 100 * progress;
        return (
          <div
            key={d.label}
            style={{ display: "flex", alignItems: "center", gap: 14, height: rowHeight }}
          >
            <div
              style={{
                width: labelWidth,
                flexShrink: 0,
                fontSize: TYPE.bodySmall,
                color: isHighlight ? C.ink100 : C.ink300,
                fontWeight: isHighlight ? 600 : 400,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {d.label}
            </div>
            <div
              style={{
                flex: 1,
                height: rowHeight - 10,
                background: "rgba(255,255,255,0.035)",
                borderRadius: 3,
                overflow: "hidden",
                minWidth: 0,
              }}
            >
              <div
                style={{
                  width: `${width}%`,
                  height: "100%",
                  background:
                    d.colour ?? (isHighlight ? C.peri300 : barColour),
                  borderRadius: 3,
                }}
              />
            </div>
            <div
              style={{
                width: 118,
                flexShrink: 0,
                textAlign: "right",
                fontSize: TYPE.bodySmall,
                fontWeight: isHighlight ? 600 : 500,
                color: isHighlight ? C.ink100 : C.ink300,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {formatValue(d.value)}
            </div>
          </div>
        );
      })}
    </div>
  );
};

/**
 * A signed bar list for a movement: positive bars grow right from a centre axis,
 * negative bars left. Used for the regional contribution to the monthly change.
 */
export const SignedBarList: React.FC<{
  data: BarDatum[];
  formatValue: (value: number) => string;
  progress?: number;
  labelWidth?: number;
  rowHeight?: number;
  highlight?: string;
  /** Distribute the rows over the container's full height. */
  fill?: boolean;
}> = ({
  data,
  formatValue,
  progress = 1,
  labelWidth = 210,
  rowHeight = 26,
  highlight,
  fill = false,
}) => {
  const max = Math.max(...data.map((d) => Math.abs(d.value)), 1);
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 11,
        ...(fill ? { height: "100%", justifyContent: "space-between", gap: undefined } : {}),
      }}
    >
      {data.map((d) => {
        const isHighlight = highlight != null && d.label === highlight;
        const pct = (Math.abs(d.value) / max) * 100 * progress;
        const positive = d.value >= 0;
        return (
          <div
            key={d.label}
            style={{ display: "flex", alignItems: "center", gap: 14, height: rowHeight }}
          >
            <div
              style={{
                width: labelWidth,
                flexShrink: 0,
                fontSize: TYPE.bodySmall,
                color: isHighlight ? C.ink100 : C.ink300,
                fontWeight: isHighlight ? 600 : 400,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {d.label}
            </div>
            <div style={{ flex: 1, position: "relative", height: rowHeight - 10, minWidth: 0 }}>
              <div
                style={{
                  position: "absolute",
                  left: "10%",
                  top: -3,
                  bottom: -3,
                  width: 1,
                  background: C.line,
                }}
              />
              <div
                style={{
                  position: "absolute",
                  left: positive ? "10%" : undefined,
                  right: positive ? undefined : "90%",
                  top: 0,
                  height: "100%",
                  width: `${pct * 0.9}%`,
                  background: isHighlight
                    ? C.peri300
                    : positive
                      ? THEME.positive
                      : THEME.negative,
                  borderRadius: 3,
                }}
              />
            </div>
            <div
              style={{
                width: 118,
                flexShrink: 0,
                textAlign: "right",
                fontSize: TYPE.bodySmall,
                fontWeight: isHighlight ? 600 : 500,
                color: isHighlight ? C.ink100 : C.ink300,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {formatValue(d.value)}
            </div>
          </div>
        );
      })}
    </div>
  );
};

/** A small multi-period line, for the funded balance series. */
export const Sparkline: React.FC<{
  points: { label: string; value: number }[];
  width: number;
  height: number;
  progress?: number;
  colour?: string;
}> = ({ points, width, height, progress = 1, colour = C.peri400 }) => {
  if (points.length < 2) return null;
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const padX = 8;
  const padY = 14;
  const coords = points.map((p, i) => {
    const x = padX + (i / (points.length - 1)) * (width - padX * 2);
    const y = height - padY - ((p.value - min) / range) * (height - padY * 2);
    return { x, y, ...p };
  });
  const shown = Math.max(2, Math.ceil(coords.length * progress));
  const visible = coords.slice(0, shown);
  const path = visible.map((c, i) => `${i === 0 ? "M" : "L"}${c.x},${c.y}`).join(" ");
  return (
    <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      <path d={path} fill="none" stroke={colour} strokeWidth={2.5} strokeLinecap="round" />
      {visible.map((c, i) => (
        <circle
          key={c.label}
          cx={c.x}
          cy={c.y}
          r={i === visible.length - 1 ? 5 : 3.5}
          fill={i === visible.length - 1 ? C.ink100 : colour}
        />
      ))}
      {coords.map((c) => (
        <text
          key={`t-${c.label}`}
          x={c.x}
          y={height + 2}
          textAnchor="middle"
          fontSize={12}
          fill={C.ink500}
          fontFamily={FONT.sans}
        >
          {c.label}
        </text>
      ))}
    </svg>
  );
};

// --------------------------------------------------------------------------- //
// Table — the product's artefact table treatment
// --------------------------------------------------------------------------- //
export const DataTable: React.FC<{
  columns: { key: string; label: string; align?: "left" | "right" }[];
  rows: Record<string, string>[];
  highlightRow?: string;
  rowsVisible?: number;
  fontSize?: number;
}> = ({ columns, rows, highlightRow, rowsVisible, fontSize = TYPE.bodySmall }) => {
  const visible = rowsVisible == null ? rows : rows.slice(0, rowsVisible);
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <div
        style={{
          display: "flex",
          gap: 20,
          paddingBottom: 9,
          borderBottom: `1px solid ${C.line}`,
        }}
      >
        {columns.map((c, i) => (
          <div
            key={c.key}
            style={{
              flex: i === 0 ? 1.6 : 1,
              fontSize: TYPE.micro,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.ink500,
              fontWeight: 600,
              textAlign: c.align ?? "left",
            }}
          >
            {c.label}
          </div>
        ))}
      </div>
      {visible.map((row, ri) => {
        const isHighlight = highlightRow != null && row[columns[0].key] === highlightRow;
        return (
          <div
            key={`${row[columns[0].key]}-${ri}`}
            style={{
              display: "flex",
              gap: 20,
              padding: "10px 0",
              borderBottom: `1px solid ${C.lineSoft}`,
              background: isHighlight ? "rgba(145,157,209,0.07)" : undefined,
            }}
          >
            {columns.map((c, i) => (
              <div
                key={c.key}
                style={{
                  flex: i === 0 ? 1.6 : 1,
                  fontSize,
                  color: isHighlight ? C.ink100 : C.ink300,
                  fontWeight: isHighlight ? 600 : 400,
                  textAlign: c.align ?? "left",
                  fontVariantNumeric: "tabular-nums",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {row[c.key]}
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
};

// --------------------------------------------------------------------------- //
// Status pill
// --------------------------------------------------------------------------- //
export const Pill: React.FC<{
  children: React.ReactNode;
  tone?: "neutral" | "positive" | "accent" | "warning";
  style?: React.CSSProperties;
}> = ({ children, tone = "neutral", style }) => {
  const tones = {
    neutral: { fg: C.ink300, bg: "rgba(255,255,255,0.05)", bd: C.line },
    positive: { fg: THEME.positive, bg: "rgba(46,125,91,0.13)", bd: "rgba(46,125,91,0.42)" },
    accent: { fg: C.peri300, bg: "rgba(145,157,209,0.13)", bd: "rgba(145,157,209,0.40)" },
    warning: { fg: THEME.rag.amber, bg: "rgba(224,169,59,0.12)", bd: "rgba(224,169,59,0.40)" },
  }[tone];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 7,
        padding: "5px 12px",
        borderRadius: 999,
        fontSize: TYPE.micro,
        fontWeight: 600,
        letterSpacing: "0.03em",
        color: tones.fg,
        background: tones.bg,
        border: `1px solid ${tones.bd}`,
        whiteSpace: "nowrap",
        ...style,
      }}
    >
      {children}
    </span>
  );
};

/** A small tick glyph, drawn rather than fonted so it renders identically. */
export const Tick: React.FC<{ size?: number; colour?: string }> = ({
  size = 13,
  colour = THEME.positive,
}) => (
  <svg width={size} height={size} viewBox="0 0 16 16" style={{ flexShrink: 0 }}>
    <path
      d="M3 8.5 L6.4 12 L13 4.5"
      fill="none"
      stroke={colour}
      strokeWidth={2.2}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

/** A right-pointing chevron, for a flow connector. */
export const Chevron: React.FC<{ size?: number; colour?: string }> = ({
  size = 16,
  colour = C.ink500,
}) => (
  <svg width={size} height={size} viewBox="0 0 16 16" style={{ flexShrink: 0 }}>
    <path
      d="M5.5 3 L11 8 L5.5 13"
      fill="none"
      stroke={colour}
      strokeWidth={1.9}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

/** A document glyph for the artefact cards. */
export const DocGlyph: React.FC<{ size?: number; colour?: string }> = ({
  size = 26,
  colour = C.peri300,
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
    <path
      d="M14 3 H7 a2 2 0 0 0 -2 2 v14 a2 2 0 0 0 2 2 h10 a2 2 0 0 0 2 -2 V8 Z"
      fill="none"
      stroke={colour}
      strokeWidth={1.5}
      strokeLinejoin="round"
    />
    <path d="M14 3 V8 H19" fill="none" stroke={colour} strokeWidth={1.5} strokeLinejoin="round" />
  </svg>
);
