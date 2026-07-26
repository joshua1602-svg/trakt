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
// <Figure> — the only way to put a computed value on screen
// --------------------------------------------------------------------------- //
/**
 * A figure the pipeline produced, in the data face with `tabular-nums`.
 *
 * The mono rule is semantic, and it is the film's signature device: if a number came
 * out of the pipeline it is set in IBM Plex Mono, and if it is Trakt talking about
 * itself it is Archivo or Inter. Routing every computed value through one component
 * means the rule cannot be broken by a scene reaching for a display token, and
 * `scripts/lint-theme.mjs` fails the build if a `<Counter>` appears outside one.
 *
 * `scale` multiplies the `stat` token — the size still originates in theme.ts.
 */
export const Figure: React.FC<{
  children: React.ReactNode;
  /** Fraction of the `stat` token size. */
  scale?: number;
  tone?: keyof typeof theme.color;
  /** Overrides `tone`; use when a scene interpolates the accent in or out. */
  color?: string;
  style?: React.CSSProperties;
}> = ({ children, scale = 1, tone = "paper", color, style }) => {
  const geometry = useGeometry();
  return (
    <div
      style={{
        ...theme.type.stat,
        fontSize: theme.type.stat.fontSize * scale * geometry.statScale,
        color: color ?? theme.color[tone],
        whiteSpace: "nowrap",
        ...style,
      }}
    >
      {children}
    </div>
  );
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
      <Figure scale={(scale ?? geometry.statScale) / geometry.statScale} color={valueColor}>
        {value}
      </Figure>
      <Rule width={ruleWidth} />
      <Stamp style={{ textAlign: align === "center" ? "center" : "left" }}>{stamp}</Stamp>
    </div>
  );
};

// --------------------------------------------------------------------------- //
// <Claim> — the display line
// --------------------------------------------------------------------------- //
/**
 * Archivo display. One per scene, maximum.
 *
 * `measure` is a FRACTION of the frame width, resolved to pixels here. A percentage
 * would be resolved against the parent, which in a shrink-to-fit flex column is the
 * claim's own content width — so a long claim ends up in a box narrower than the frame
 * and sits off-centre. Resolving against the layout makes the measure definite.
 */
export const Claim: React.FC<{
  children: React.ReactNode;
  align?: "left" | "center";
  /** Full-strength `signal`. Only the value being proven in that scene carries it. */
  accent?: boolean;
  /** Fraction of the frame width the claim may occupy. */
  measure?: number;
  style?: React.CSSProperties;
}> = ({ children, align = "center", accent = false, measure = 0.72, style }) => {
  const geometry = useGeometry();
  return (
    <div
      style={{
        ...theme.type.display,
        fontSize: theme.type.display.fontSize * geometry.displayScale,
        color: accent ? theme.color.signal : theme.color.paper,
        textAlign: align,
        width: geometry.width * measure,
        // Even line lengths rather than a one-word last line. Display type is the one
        // place a widow is genuinely expensive.
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
  /** Fraction of the frame width, as for <Claim>. */
  measure?: number;
}> = ({ children, align = "center", measure = 0.8 }) => {
  const geometry = useGeometry();
  return (
    <div
      style={{
        ...theme.type.headline,
        fontSize: theme.type.headline.fontSize * geometry.displayScale,
        color: theme.color.paper,
        textAlign: align,
        width: geometry.width * measure,
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
/**
 * A governed output.
 *
 * The TITLE is what the artefact is, in plain English and in the body face — that is
 * what a COO reads. The filename drops into the mono meta line beneath, with the
 * figures, because a filename is a source identifier and belongs in the data face. The
 * old arrangement made `platform_canonical_typed.csv` the largest text on its own card,
 * which asked the viewer to care about a path.
 *
 * `scope` says which book the artefact is an output of — SPV1, PLATFORM, or ALL. It sits
 * above the title because "which portfolio is this about" is the first question anyone
 * asks of a governed output, and a card that answers it cannot be misread as speaking
 * for the whole business. The scope word itself carries `signal` at the soft strength;
 * everything after the separator is `mute`, so the eye picks the scope out of a wall of
 * six cards without any of them shouting.
 *
 * `confirm` is a check state ("XSD validated") and is the one other thing on the card
 * that carries `signal`, at the same soft strength — never full.
 */
export const ArtifactCard: React.FC<{
  title: string;
  /** Which book this output covers. Omitted only where scope is unambiguous. */
  scope?: string;
  /** Trailing text on the scope line, after the separator. */
  scopeNote?: string;
  meta: string[];
  confirm?: string;
  style?: React.CSSProperties;
}> = ({ title, scope, scopeNote, meta, confirm, style }) => {
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
      {scope ? (
        <Stamp>
          <span style={{ color: theme.color.signal, opacity: theme.signalSoftOpacity }}>
            {scope}
          </span>
          {scopeNote ? ` · ${scopeNote}` : null}
        </Stamp>
      ) : null}
      <Body style={{ fontSize: theme.type.body.fontSize * (isSquare ? 0.82 : 0.92) }}>
        {title}
      </Body>
      <Rule />
      {meta.map((line) => (
        <Stamp key={line} style={{ overflowWrap: "anywhere" }}>
          {line}
        </Stamp>
      ))}
      {confirm ? (
        <Stamp tone="signal" style={{ opacity: theme.signalSoftOpacity }}>
          {confirm}
        </Stamp>
      ) : null}
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
