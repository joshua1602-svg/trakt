"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import { cx } from "@/components/ui";

/**
 * Scroll-triggered reveal: a 14px rise over 240ms ease-out, staggered across
 * siblings via `delay`. Fires once and never replays on scroll-up.
 *
 * The hidden initial state is applied only when BOTH hold: the `js` class is
 * on <html> (set inline in the layout, so no-JS visitors always see final
 * state) and the visitor has not asked for reduced motion — see the
 * `[data-reveal]` rules in `app/globals.css`. Reduced-motion visitors get the
 * final state with no transform at any point.
 */
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
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      const id = window.setTimeout(() => setRevealed(true), 0);
      return () => window.clearTimeout(id);
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setRevealed(true);
            observer.disconnect();
          }
        }
      },
      { threshold: 0.15, rootMargin: "0px 0px -5% 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      data-reveal
      className={cx(className, revealed && "is-revealed")}
      style={{ "--reveal-delay": `${delay}ms` } as React.CSSProperties}
    >
      {children}
    </div>
  );
}
