"use client";

import { useEffect, useRef, useState } from "react";

import { ControlPreview } from "@/components/site/Content";

/**
 * The Risk & Controls demo loop: an 18-second muted autoplay video rendered
 * from the Remotion composition in `demo-video/src/landing/ControlsDemo.tsx`,
 * showing document → structured requirement → reviewed control → live and
 * forward monitoring.
 *
 * Performance and accessibility posture:
 *
 * - The server renders the poster image only; the video element mounts the
 *   first time the section approaches the viewport (IntersectionObserver,
 *   200px margin), so the ~1–3 MB asset never competes with initial load or
 *   LCP. A fixed aspect ratio means mounting the video shifts no layout.
 * - `prefers-reduced-motion` visitors get the static `ControlPreview`
 *   instead — real DOM text, not a frozen frame — and the same happens if
 *   the video fails to load. The page works with no motion at all.
 * - Playback pauses whenever the section leaves the viewport.
 * - The sequence is described for assistive technology on the wrapper; the
 *   visible caption below carries the illustrative-figures provenance in DOM
 *   text, mirroring the stamp burned into the video.
 */

/**
 * Two encodings of the same render: VP9 first for Chromium and Firefox
 * (including codec-stripped Chromium builds that ship no H.264), H.264 for
 * Safari and iOS. The browser takes the first source it can decode.
 */
const WEBM_SRC = "/controls-demo.webm";
const MP4_SRC = "/controls-demo.mp4";
const POSTER_SRC = "/controls-demo-poster.png";

const DESCRIPTION =
  "Demonstration: clauses in a portfolio covenant schedule extract are " +
  "identified and structured into proposed controls, reviewed and activated " +
  "by a person, then monitored against the funded book, the expected " +
  "forecast and the full pipeline, ending on a projected breach horizon.";

export function ControlsDemoLoop() {
  const [reducedMotion, setReducedMotion] = useState(false);
  const [failed, setFailed] = useState(false);
  const [active, setActive] = useState(false);
  const holderRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const media = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!media) return;
    const update = () => setReducedMotion(media.matches);
    update();
    media.addEventListener?.("change", update);
    return () => media.removeEventListener?.("change", update);
  }, []);

  useEffect(() => {
    if (reducedMotion || failed) return;
    const node = holderRef.current;
    if (!node) return;
    if (typeof IntersectionObserver === "undefined") {
      // No observer support: mount the video on the next tick instead of
      // synchronously inside the effect body.
      const id = window.setTimeout(() => setActive(true), 0);
      return () => window.clearTimeout(id);
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActive(true);
          const video = videoRef.current;
          if (!video) continue;
          if (entry.isIntersecting) void video.play().catch(() => {});
          else video.pause();
        }
      },
      { rootMargin: "200px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [reducedMotion, failed]);

  if (reducedMotion || failed) {
    return <ControlPreview />;
  }

  return (
    <div>
      <div
        ref={holderRef}
        role="img"
        aria-label={DESCRIPTION}
        className="aspect-[5/4] overflow-hidden rounded-2xl border border-line bg-navy-900/80 shadow-[0_24px_60px_-30px_rgba(0,0,0,0.9)]"
      >
        {active ? (
          <video
            ref={videoRef}
            className="h-full w-full object-cover"
            poster={POSTER_SRC}
            autoPlay
            muted
            loop
            playsInline
            preload="none"
            aria-hidden="true"
            tabIndex={-1}
            onError={() => {
              // Fires on the <video> for fatal playback errors; source-list
              // exhaustion is caught on the last <source> below.
              if (videoRef.current?.error) setFailed(true);
            }}
          >
            <source src={WEBM_SRC} type="video/webm" />
            {/* The last source's error event means no source was playable. */}
            <source src={MP4_SRC} type="video/mp4" onError={() => setFailed(true)} />
          </video>
        ) : (
          // Server-rendered state: the poster alone, so nothing heavy loads
          // before the section is near the viewport. The wrapper carries the
          // description, so the image itself is decorative.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={POSTER_SRC}
            alt=""
            aria-hidden="true"
            className="h-full w-full object-cover"
            loading="lazy"
            decoding="async"
          />
        )}
      </div>
      <p className="mt-3 text-[11px] leading-relaxed text-ink-500">
        18-second loop: covenant clauses become reviewed, activated controls —
        monitored against the funded book, the forecast and the pipeline. Figures
        illustrative.
      </p>
    </div>
  );
}
