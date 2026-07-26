/**
 * The component contract. Every scene is composed from these and nothing else, so
 * consistency is enforced by construction rather than by discipline.
 *
 *   <Stat>          a computed figure: mono, tabular-nums, always with its rule
 *   <Claim>         the one display line a scene is allowed
 *   <ArtifactCard>  a governed output, fixed dimensions
 *   <Counter>       the only count-up in the film
 *   <Chrome>        lockup and disclaimer, rendered once outside <Series>
 *
 * Supporting pieces used by more than one scene live here too — `Enter`, `Label`,
 * `Rule`, `Stamp`, `Panel` — for the same reason.
 *
 * No colour, size, weight or duration is declared in this file. Everything comes
 * from src/theme.ts.
 */

import React from "react";
import {
  interpolate,
  interpolateColors,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

import theme, { type Layout } from "../theme";

// --------------------------------------------------------------------------- //
// Layout context
// --------------------------------------------------------------------------- //
const LayoutContext = React.createContext<Layout>("wide");

export const LayoutProvider: React.FC<{ layout: Layout; children: React.ReactNode }> = ({
  layout,
  children,
}) => <LayoutContext.Provider value={layout}>{children}</LayoutContext.Provider>;

export const useLayout = (): Layout => React.useContext(LayoutContext);

/** The geometry tokens for the active layout. */
export const useGeometry = () => theme.layout[useLayout()];

/** True in the 1080x1080 crop, where stacks go vertical and copy tightens. */
export const useIsSquare = (): boolean => useLayout() === "square";

// --------------------------------------------------------------------------- //
// The single entrance
// --------------------------------------------------------------------------- //
/**
 * Opacity 0 -> 1 with a 12px rise, on the film's one spring. Nothing scales in and
 * nothing bounces, so this is the only entrance any element gets.
 */
export const Enter: React.FC<{
  /** Frames from the start of the enclosing sequence. */
  at?: number;
  /** Index in a staggered group. */
  index?: number;
  /** Frames the element stays up; omitted means "until the sequence ends". */
  hold?: number;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ at = 0, index = 0, hold, style, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const start = at + index * theme.motion.stagger;
  const progress = spring({
    frame: frame - start,
    fps,
    config: theme.motion.spring,
    durationInFrames: theme.motion.base,
  });
  const out =
    hold === undefined
      ? 1
      : interpolate(frame, [start + hold - theme.motion.quick, start + hold], [1, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
  return (
    <div
      style={{
        ...style,
        opacity: progress * out,
        transform: `translateY(${(1 - progress) * theme.motion.enterRise}px)`,
      }}
    >
      {children}
    </div>
  );
};

// --------------------------------------------------------------------------- //
// Type primitives
// --------------------------------------------------------------------------- //
export const Label: React.FC<{
  children: React.ReactNode;
  tone?: keyof typeof theme.color;
  style?: React.CSSProperties;
}> = ({ children, tone = "mute", style }) => (
  <div style={{ ...theme.type.label, color: theme.color[tone], ...style }}>{children}</div>
);

export const Stamp: React.FC<{
  children: React.ReactNode;
  tone?: keyof typeof theme.color;
  style?: React.CSSProperties;
}> = ({ children, tone = "mute", style }) => (
  <div style={{ ...theme.type.stamp, color: theme.color[tone], ...style }}>{children}</div>
);

export const Body: React.FC<{
  children: React.ReactNode;
  tone?: keyof typeof theme.color;
  style?: React.CSSProperties;
}> = ({ children, tone = "paper", style }) => (
  <div style={{ ...theme.type.body, color: theme.color[tone], ...style }}>{children}</div>
);

/** A 1px hairline. The only depth cue in the film. */
export const Rule: React.FC<{ width?: number | string; tone?: keyof typeof theme.color }> = ({
  width = "100%",
  tone = "rule",
}) => (
  <div
    style={{ width, height: theme.hairline, backgroundColor: theme.color[tone], flexShrink: 0 }}
  />
);

/** A surface. One elevation, a 1px border, no shadow and no gradient. */
export const Panel: React.FC<{
  children?: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ children, style }) => (
  <div
    style={{
      backgroundColor: theme.color.hull,
      border: `${theme.hairline}px solid ${theme.color.rule}`,
      borderRadius: theme.radius.card,
      ...style,
    }}
  >
    {children}
  </div>
);

// --------------------------------------------------------------------------- //
// <Counter> — the only count-up
// --------------------------------------------------------------------------- //
/**
 * Wraps `spring` + `interpolate`. Never hand-roll a count-up: the digits have to
 * ride the film's one spring, and `tabular-nums` has to be on or they jitter.
 */
export const Counter: React.FC<{
  from: number;
  to: number;
  /** Frames from the start of the enclosing sequence at which the count begins. */
  at?: number;
  /** Frames the count takes. Defaults to the `slow` token. */
  frames?: number;
  format: (value: number) => string;
  style?: React.CSSProperties;
}> = ({ from, to, at = 0, frames = theme.motion.slow, format, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({
    frame: frame - at,
    fps,
    config: theme.motion.spring,
    durationInFrames: frames,
  });
  return <span style={style}>{format(from + (to - from) * progress)}</span>;
};

// --------------------------------------------------------------------------- //
// <Stat> — a computed figure with its provenance rule
// --------------------------------------------------------------------------- //
/**
 * The film's signature. Every computed figure sits above a 1px hairline with a
 * mono `stamp` line beneath it naming where it came from:
 *
 *     £1.96bn
 *     ──────────────────────────────
 *     ALP_ORIGINATION + ALP_ACQUIRED · 2026-06-30
 *
 * `stamp` is required — there is no way to render a governed figure without saying
 * where it came from.
 */
export const Stat: React.FC<{
  /** Already formatted, or a <Counter>. */
  value: React.ReactNode;
  stamp: string;
  /**
   * Set only on the ONE element per frame that carries the accent. A number
   * between 0 and 1 hands the accent over gradually, which is how a scene passes
   * `signal` from one element to the next without ever having two at once.
   */
  accent?: boolean | number;
  /**
   * Normally omitted: the block sizes to its widest child (which is the stamp), so
   * the rule spans the provenance line exactly. Pass a width only to force a
   * narrower rule inside a panel.
   */
  ruleWidth?: number | string;
  align?: "left" | "center";
  scale?: number;
  style?: React.CSSProperties;
}> = ({ value, stamp, accent = false, ruleWidth = "100%", align = "left", scale, style }) => {
  const geometry = useGeometry();
  const size = theme.type.stat.fontSize * (scale ?? geometry.statScale);
  const accentAmount = accent === true ? 1 : accent === false ? 0 : accent;
  const valueColor = interpolateColors(
    accentAmount,
    [0, 1],
    [theme.color.paper, theme.color.signal],
  );
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: align === "center" ? "center" : "flex-start",
        gap: theme.motion.quick,
        // Shrink-wrap, so the hairline is exactly as wide as the stamp beneath it
        // rather than overhanging it or being overhung by it.
        width: "fit-content",
        ...style,
      }}
    >
      <div
        style={{
          ...theme.type.stat,
          fontSize: size,
          color: valueColor,
          whiteSpace: "nowrap",
        }}
      >
        {value}
      </div>
      <Rule width={ruleWidth} />
      <Stamp style={{ textAlign: align === "center" ? "center" : "left" }}>{stamp}</Stamp>
    </div>
  );
};

// --------------------------------------------------------------------------- //
// <Claim> — the display line
// --------------------------------------------------------------------------- //
/** Archivo display. One per scene, maximum. */
export const Claim: React.FC<{
  children: React.ReactNode;
  align?: "left" | "center";
  /** The close is the film's only accented claim. */
  accent?: boolean;
  maxWidth?: number | string;
  style?: React.CSSProperties;
}> = ({ children, align = "center", accent = false, maxWidth = "72%", style }) => {
  const geometry = useGeometry();
  return (
    <div
      style={{
        ...theme.type.display,
        fontSize: theme.type.display.fontSize * geometry.displayScale,
        color: accent ? theme.color.signal : theme.color.paper,
        textAlign: align,
        maxWidth,
        // Even line lengths rather than a one-word last line. Display type is the
        // one place a widow is genuinely expensive.
        textWrap: "balance",
        ...style,
      }}
    >
      {children}
    </div>
  );
};

/** The headline size, for the one place a claim has to sit inside a stack. */
export const Headline: React.FC<{
  children: React.ReactNode;
  align?: "left" | "center";
  maxWidth?: number | string;
}> = ({ children, align = "center", maxWidth = "80%" }) => {
  const geometry = useGeometry();
  return (
    <div
      style={{
        ...theme.type.headline,
        fontSize: theme.type.headline.fontSize * geometry.displayScale,
        color: theme.color.paper,
        textAlign: align,
        maxWidth,
        textWrap: "balance",
      }}
    >
      {children}
    </div>
  );
};

// --------------------------------------------------------------------------- //
// <ArtifactCard> — a governed output
// --------------------------------------------------------------------------- //
export const ArtifactCard: React.FC<{
  filename: string;
  meta: string[];
  style?: React.CSSProperties;
}> = ({ filename, meta, style }) => {
  const isSquare = useIsSquare();
  return (
    <Panel
      style={{
        width: isSquare ? 320 : 380,
        height: "100%",
        padding: theme.motion.base + theme.motion.quick / 2,
        display: "flex",
        flexDirection: "column",
        gap: theme.motion.quick,
        justifyContent: "flex-start",
        ...style,
      }}
    >
      <div
        style={{
          ...theme.type.stamp,
          color: theme.color.paper,
          overflowWrap: "anywhere",
        }}
      >
        {filename}
      </div>
      <Rule />
      {meta.map((line) => (
        <Stamp key={line}>{line}</Stamp>
      ))}
    </Panel>
  );
};

// --------------------------------------------------------------------------- //
// <Chrome> — lockup and disclaimer
// --------------------------------------------------------------------------- //
/**
 * The Trakt lockup, one fixed size for the whole film, and the synthetic-data
 * disclaimer. Rendered once, OUTSIDE the <Series>, so branding structurally cannot
 * drift between scenes and the disclaimer is present in every single frame.
 *
 * `hideLockup` is set only for the close, where the lockup animates to centre — the
 * one time chrome moves. The disclaimer never moves.
 */
export const Lockup: React.FC = () => {
  const { markSize, markStroke, gap } = theme.lockup;
  const inset = markStroke / 2;
  return (
    <div style={{ display: "flex", alignItems: "center", gap }}>
      <svg
        width={markSize}
        height={markSize}
        viewBox={`0 0 ${markSize} ${markSize}`}
        aria-hidden
        style={{ display: "block" }}
      >
        <rect
          x={inset}
          y={inset}
          width={markSize - markStroke}
          height={markSize - markStroke}
          fill="none"
          stroke={theme.color.signal}
          strokeWidth={markStroke}
        />
        <path
          d={`M${markSize * 0.23} ${markSize * 0.33}H${markSize * 0.77}`}
          stroke={theme.color.signal}
          strokeWidth={markStroke}
        />
        <path
          d={`M${markSize / 2} ${markSize * 0.33}V${markSize * 0.73}`}
          stroke={theme.color.signal}
          strokeWidth={markStroke}
        />
      </svg>
      <div
        style={{
          ...theme.type.headline,
          fontSize: theme.lockup.wordSize,
          letterSpacing: "0.02em",
          color: theme.color.paper,
        }}
      >
        trakt
      </div>
    </div>
  );
};

export const Chrome: React.FC<{ notice: string; hideLockup?: boolean }> = ({
  notice,
  hideLockup = false,
}) => {
  const geometry = useGeometry();
  return (
    <>
      {hideLockup ? null : (
        <div style={{ position: "absolute", top: geometry.edge, left: geometry.edge }}>
          <Lockup />
        </div>
      )}
      <div style={{ position: "absolute", bottom: geometry.edge, left: geometry.edge }}>
        <Label>{notice}</Label>
      </div>
    </>
  );
};
