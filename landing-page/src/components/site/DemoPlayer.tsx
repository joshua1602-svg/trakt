"use client";

import { useCallback, useRef, useState, type ReactNode } from "react";

import { buttonStyles, cx } from "@/components/ui";

/**
 * The controlled demo player: poster first, motion only on request.
 *
 * States: idle (poster + play overlay) → playing ⇄ paused → ended
 * ("Watch again" overlay). The video never autoplays, never loops, and with
 * `preload="none"` costs nothing until the visitor presses play — so nobody
 * arrives mid-animation, and reduced-motion visitors see no motion they did
 * not ask for. If no source is playable, the DOM fallback renders instead.
 */

type PlayerState = "idle" | "playing" | "paused" | "ended";

export function DemoPlayer({
  overlayLabel,
  durationLabel,
  poster,
  webmSrc,
  mp4Src,
  description,
  caption,
  fallback,
  plateId,
  platePosition = "center",
}: {
  /** The play-overlay button text, e.g. "Watch controls demo". */
  overlayLabel: string;
  /** Shown beside the overlay button, e.g. "~18 sec". */
  durationLabel?: string;
  poster: string;
  webmSrc: string;
  mp4Src: string;
  /** Assistive description of what the demo shows. */
  description: string;
  /** One short line under the frame. */
  caption?: string;
  /** Rendered instead of the player when no source is playable. */
  fallback: ReactNode;
  /**
   * Identifies this player's play plate to the geometry guards. There is more
   * than one player on the page now, so a shared hook would resolve to every
   * plate at once and the guard would measure whichever came first.
   */
  plateId: string;
  /**
   * Where the play plate sits in the frame.
   *
   * `center` is the default and is what the controls poster is drawn for — its
   * monitoring card is lifted into the upper third so a centred plate clears
   * it. `lower` exists for the A2A player, whose poster is a real frame of the
   * film (frame 1210) rather than a composition arranged around the plate. The
   * film draws everything above 65% of the frame height, so the plate drops
   * into the empty band beneath it and the still can be the genuine article.
   * Getting a poster that the film never shows was the alternative, and it is
   * what shipped for one pass.
   */
  platePosition?: "center" | "lower";
}) {
  const [state, setState] = useState<PlayerState>("idle");
  const [failed, setFailed] = useState(false);
  const [progress, setProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  const play = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    void video.play().catch(() => setFailed(true));
  }, []);

  const pause = useCallback(() => videoRef.current?.pause(), []);

  const replay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    void video.play().catch(() => setFailed(true));
  }, []);

  if (failed) return <>{fallback}</>;

  return (
    <figure className="m-0 max-w-4xl">
      <div className="relative aspect-[5/4] overflow-hidden rounded-2xl border border-line bg-navy-900/80 shadow-[0_24px_60px_-30px_rgba(0,0,0,0.9)]">
        <video
          ref={videoRef}
          className="h-full w-full object-cover"
          poster={poster}
          preload="none"
          muted
          playsInline
          aria-label={description}
          onPlay={() => setState("playing")}
          onPause={() => setState((s) => (s === "ended" ? s : "paused"))}
          onEnded={() => setState("ended")}
          onTimeUpdate={() => {
            const video = videoRef.current;
            if (video?.duration) setProgress(video.currentTime / video.duration);
          }}
          onError={() => {
            if (videoRef.current?.error) setFailed(true);
          }}
        >
          <source src={webmSrc} type="video/webm" />
          {/* The last source's error event means no source was playable. */}
          <source src={mp4Src} type="video/mp4" onError={() => setFailed(true)} />
        </video>

        {state === "idle" ? (
          /* The still stays at full contrast and sharpness — a global overlay
             made working software look disabled — and the plate is fully
             opaque, because live text ghosting through it read as a rendering
             fault. Centred on the FRAME, not the figure: the figure includes
             the caption, which pushed the centre down by half its height. The
             poster is drawn for a centred plate, with the concentration card
             lifted into the upper third so the three rows and the breach
             horizon stay clear (demo-video/src/landing/ControlsPoster.tsx). */
          <div
            className={cx(
              "absolute flex justify-center",
              platePosition === "lower"
                ? /* Anchored by FRACTION of the frame, not by padding from the
                     bottom. The plate is a fixed height while the frame scales
                     with viewport width, so a percentage inset from the bottom
                     pushed the plate's top edge up to 56% of a 280px frame at
                     390 — back over the findings the move was meant to clear.
                     Fixing the top edge at 68% holds at every width, and the
                     plate still lands inside the frame at the narrowest. */
                  "inset-x-0 top-[66%]"
                : "inset-0 items-center",
            )}
          >
            {/* The hook sits on the visible plate, not the full-size
                positioning container — measuring the container told the guard
                the plate filled the frame. */}
            <div
              data-plate={plateId}
              className={cx(
                "flex flex-col items-center rounded-2xl border border-line bg-navy-950 shadow-[0_18px_40px_-20px_rgba(0,0,0,0.95)]",
                /* The lower plate is tighter because it has to fit in what the
                   poster leaves: at 390 the frame is only 280px tall and the
                   band below the film's content is under 100px. The centred
                   plate has the whole frame and keeps its original padding. */
                platePosition === "lower" ? "gap-1.5 px-5 py-3" : "gap-2 px-6 py-5",
              )}
            >
              <button type="button" onClick={play} className={buttonStyles.primary}>
                <PlayGlyph /> {overlayLabel}
              </button>
              {durationLabel ? (
                <span className="text-[12px] font-medium text-ink-300">{durationLabel}</span>
              ) : null}
            </div>
          </div>
        ) : null}

      </div>

      {/* The transport sits BELOW the frame, not across the bottom of it.
          Overlaid, it stood on the last rows of the film for the whole run —
          the A2A demo draws to the frame edge, so its closing lines were
          simply not readable. Auto-hiding is the usual answer and it is a bad
          one here: on a phone there is no hover, so the controls come back
          only by tapping the thing you are trying to watch, and a pause
          control you cannot find is worse than one that costs 40px.

          "Watch again" is in this strip too, for the same reason. As a
          centred button on a scrim over the frame it covered the film's last
          frame — which, for a demo that ends on its assessment, is the frame
          most worth reading. Ending now leaves the picture alone. */}
      {state === "idle" ? null : (
        <div className="mt-2 flex items-center gap-2">
          {state === "ended" ? (
            <button
              type="button"
              onClick={replay}
              className={cx(buttonStyles.secondary, "shrink-0")}
            >
              <ReplayGlyph /> Watch again
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={state === "playing" ? pause : play}
                aria-label={state === "playing" ? "Pause demo" : "Resume demo"}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-line bg-navy-850 text-ink-100 hover:border-peri-500"
              >
                {state === "playing" ? <PauseGlyph /> : <PlayGlyph />}
              </button>
              <button
                type="button"
                onClick={replay}
                aria-label="Restart demo from the beginning"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-line bg-navy-850 text-ink-100 hover:border-peri-500"
              >
                <ReplayGlyph />
              </button>
            </>
          )}
          <div
            role="progressbar"
            aria-label="Demo progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progress * 100)}
            className="h-1 flex-1 overflow-hidden rounded-full bg-navy-800"
          >
            <div
              className="h-full rounded-full bg-peri-400"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}

      {caption ? (
        <figcaption className="mt-3 text-[11px] leading-relaxed text-ink-500">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

function PlayGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M7 4l14 8-14 8z" />
    </svg>
  );
}

function PauseGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M6 4h4v16H6zM14 4h4v16h-4z" />
    </svg>
  );
}

function ReplayGlyph() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5" />
    </svg>
  );
}
