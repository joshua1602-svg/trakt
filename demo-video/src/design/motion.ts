/**
 * Motion vocabulary — restrained, institutional, deterministic.
 *
 * Every helper is a pure function of the frame number, so a rendered frame is
 * identical on every machine and every run. Nothing here uses time, randomness
 * or measured layout.
 */

import { interpolate, spring, type SpringConfig } from "remotion";

export const FPS = 30;

/** Seconds → frames. Used everywhere so timings read in seconds. */
export const s = (seconds: number): number => Math.round(seconds * FPS);

/**
 * The house easing: a slow-out cubic. Fast enough to feel responsive, slow
 * enough at the tail to read as considered rather than snappy.
 */
export const EASE_OUT = (t: number): number => 1 - Math.pow(1 - t, 3);
export const EASE_IN_OUT = (t: number): number =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/** A soft spring for elements that should settle rather than stop. */
export const SOFT_SPRING: Partial<SpringConfig> = {
  damping: 200,
  mass: 0.7,
  stiffness: 100,
};

/**
 * Fade-and-rise entrance. The 8px rise is deliberate: enough to give direction,
 * small enough that a still frame never looks mid-animation.
 */
export const enter = (
  frame: number,
  delay = 0,
  duration = s(0.6),
  rise = 8,
): { opacity: number; transform: string } => {
  const t = interpolate(frame - delay, [0, duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: EASE_OUT,
  });
  return {
    opacity: t,
    transform: `translateY(${(1 - t) * rise}px)`,
  };
};

/** Fade only, for text that should not move (captions, figures in a tile). */
export const fade = (frame: number, delay = 0, duration = s(0.45)): number =>
  interpolate(frame - delay, [0, duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: EASE_OUT,
  });

/** Fade in, hold, fade out — for a caption with a known lifetime. */
export const fadeInOut = (
  frame: number,
  start: number,
  duration: number,
  ramp = s(0.4),
): number =>
  interpolate(
    frame,
    [start, start + ramp, start + duration - ramp, start + duration],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: EASE_OUT },
  );

/**
 * A slow push-in. Institutional camera language: barely perceptible, enough to
 * stop a static composition from feeling like a slide.
 */
export const pushIn = (
  frame: number,
  duration: number,
  from = 1.0,
  to = 1.035,
): string => {
  const scale = interpolate(frame, [0, duration], [from, to], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: EASE_IN_OUT,
  });
  return `scale(${scale})`;
};

/** A vertical drift, for a full-height panel that should feel alive. */
export const drift = (frame: number, duration: number, pixels = 24): string => {
  const y = interpolate(frame, [0, duration], [0, -pixels], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: EASE_IN_OUT,
  });
  return `translateY(${y}px)`;
};

/**
 * Count a figure up to its final value. Only ever used for a value that is
 * already fixed by the fixtures — the animation is presentational, the number
 * is not.
 */
export const countTo = (
  frame: number,
  target: number,
  delay = 0,
  duration = s(1.1),
): number => {
  const t = interpolate(frame - delay, [0, duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: EASE_OUT,
  });
  return target * t;
};

/** A spring-settled progress value in [0, 1], for a bar or a connector. */
export const grow = (frame: number, fps: number, delay = 0): number =>
  spring({ frame: frame - delay, fps, config: SOFT_SPRING, durationInFrames: s(0.9) });

/**
 * A stagger helper: the delay for item `index` in a list.
 * Kept small — a long stagger reads as a slideshow.
 */
export const stagger = (index: number, step = s(0.09)): number => index * step;

/**
 * Typewriter reveal for a chat message. Returns the substring to show.
 * Character-accurate and frame-deterministic.
 */
export const typed = (
  text: string,
  frame: number,
  delay: number,
  charsPerFrame = 2.6,
): string => {
  if (frame < delay) return "";
  const chars = Math.floor((frame - delay) * charsPerFrame);
  return text.slice(0, Math.max(0, Math.min(text.length, chars)));
};

/** Frames a typewriter reveal of `text` needs, plus a settle beat. */
export const typedDuration = (text: string, charsPerFrame = 2.6): number =>
  Math.ceil(text.length / charsPerFrame) + s(0.2);
