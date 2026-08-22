"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import { cx } from "@/components/ui";

/**
 * Scroll-triggered reveal — an enhancement layered onto readable content,
 * never a precondition for it.
 *
 * The server renders `data-reveal="static"`, which no opacity rule matches,
 * so the page is fully readable before any JavaScript runs, with a blocked
 * or broken bundle, under `prefers-reduced-motion`, in print, and in any
 * capture tool. Only after mount, and only for content more than a viewport
 * below the fold, does an element flip to `pending` (hidden) — by definition
 * off-screen, so nothing a reader is looking at ever disappears.
 *
 * The failsafe is per-element and armed *before* the observer is
 * constructed, so a construction failure still resolves to visible. An
 * IntersectionObserver always delivers an initial callback for an observed
 * element; receiving one is proof the observer works, and cancels the
 * failsafe so the scroll reveal can happen whenever the reader arrives.
 * If that callback never comes, the failsafe reveals the content anyway.
 *
 * A second failsafe covers the case the first cannot: an element that is
 * jumped PAST rather than scrolled through. It reveals on contact or on being
 * left behind, so nothing the reader has already gone by stays hidden.
 */

type RevealState = "static" | "pending" | "revealed";

/** How far below the fold content must sit before it is animated at all. */
const OFFSCREEN_MARGIN_VIEWPORTS = 2;

/** Ceiling on how long content may stay hidden if the observer never reports. */
const FAILSAFE_MS = 1500;

export function Reveal({
  delay = 0,
  className,
  children,
}: {
  /** Milliseconds of stagger relative to siblings (multiples of 60). */
  delay?: number;
  className?: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<RevealState>("static");

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    // Reduced motion: never hide, never animate.
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;

    // Anything on screen, partially on screen, above the fold, or within a
    // viewport of it stays static — so a reader scrolling during hydration
    // never sees content appear, vanish, then reappear.
    const { top } = node.getBoundingClientRect();
    if (top < window.innerHeight * OFFSCREEN_MARGIN_VIEWPORTS) return;

    let cancelled = false;
    const reveal = () => {
      if (!cancelled) setState("revealed");
    };

    // Deferred a tick so no state is set synchronously in the effect body.
    const armId = window.setTimeout(() => {
      if (!cancelled) setState("pending");
    }, 0);

    // Armed before the observer exists, so a throwing constructor or a
    // missing implementation still ends with the content visible.
    let failsafeId: number | undefined = window.setTimeout(reveal, FAILSAFE_MS);
    const cancelFailsafe = () => {
      if (failsafeId !== undefined) {
        window.clearTimeout(failsafeId);
        failsafeId = undefined;
      }
    };

    let observer: IntersectionObserver | undefined;
    try {
      observer = new IntersectionObserver(
        (entries) => {
          // Any callback at all proves the observer is live.
          cancelFailsafe();
          for (const entry of entries) {
            // Revealed on contact, and ALSO if the reader has already gone
            // past it. An instant scroll — an anchor jump, a restored scroll
            // position, find-in-page — moves an element from below the
            // viewport to above it in a single frame, so it never intersects
            // and, before this, stayed hidden for the rest of the session.
            // Content the reader has scrolled past must never be invisible;
            // that is the whole point of the failsafe, and this is the case
            // the failsafe missed, because the observer's first callback had
            // already cancelled it.
            const scrolledPast = entry.boundingClientRect.bottom <= 0;
            if (entry.isIntersecting || scrolledPast) {
              reveal();
              observer?.disconnect();
            }
          }
        },
        // The root is EXPANDED at the bottom, never shrunk. A negative margin
        // here creates a dead band: an element resting just inside the
        // viewport's lower edge is readable but not "intersecting", so it
        // stays hidden. Threshold 0 for the same reason — any contact
        // reveals. The stagger supplies the polish; the threshold must not.
        //
        // The TOP is expanded without limit, which is what makes "already
        // scrolled past" reveal. An observer only reports when intersection
        // CHANGES, so an element jumped over — an anchor, a restored scroll
        // position, find-in-page — went from not-intersecting-below to
        // not-intersecting-above without ever reporting, and stayed hidden
        // for the rest of the session. With the root running up past the top
        // of the document, everything behind the reader is inside it, so the
        // jump is a change the observer does report.
        { threshold: 0, rootMargin: "999999px 0px 10% 0px" },
      );
      observer.observe(node);
    } catch {
      // No observer: the failsafe above is the whole mechanism.
    }

    return () => {
      cancelled = true;
      window.clearTimeout(armId);
      cancelFailsafe();
      observer?.disconnect();
    };
  }, []);

  return (
    <div
      ref={ref}
      data-reveal={state}
      className={className ? cx(className) : undefined}
      style={{ "--reveal-delay": `${delay}ms` } as React.CSSProperties}
    >
      {children}
    </div>
  );
}
