"use client";

import { useState } from "react";

import { Card, cx } from "@/components/ui";
import { track } from "@/lib/analytics";
import { publicConfig } from "@/lib/public-config";

/**
 * The product overview video.
 *
 * ── No final video asset exists in this repository ──────────────────────────
 * A search of the repo found no `.mp4`/`.webm`/`.mov` and no hosted video URL
 * in any configuration, so this component ships as a documented placeholder:
 *
 *   • Set `NEXT_PUBLIC_DEMO_VIDEO_URL` to the final hosted asset (an absolute
 *     https URL on a domain you control, or a path under `public/`), and
 *     optionally `NEXT_PUBLIC_DEMO_VIDEO_POSTER` for the poster frame.
 *   • With neither set, the section renders the fallback below and the page is
 *     complete without it — nothing 404s, and no local development path is
 *     baked into the build.
 *   • See `landing-page/README.md` § "Replacing the demo video".
 *
 * The player never autoplays, always shows native controls, and is preceded by
 * a poster so it costs no bandwidth until the visitor asks for it.
 */
export function DemoVideo() {
  const [failed, setFailed] = useState(false);
  const src = publicConfig.videoUrl.trim();
  const poster = publicConfig.videoPoster.trim();

  if (!src || failed) return <VideoFallback unavailable={failed} />;

  return (
    <Card className="overflow-hidden p-0">
      <video
        className="aspect-video w-full bg-navy-950"
        controls
        preload="none"
        playsInline
        poster={poster || undefined}
        onPlay={() => track("demo_video_play")}
        onError={() => setFailed(true)}
      >
        <source src={src} />
        Your browser cannot play this video. The interactive demonstration below
        covers the same ground.
      </video>
    </Card>
  );
}

function VideoFallback({ unavailable }: { unavailable: boolean }) {
  return (
    <Card
      className={cx(
        "flex flex-col items-center justify-center gap-3 py-14 text-center",
        "bg-[linear-gradient(180deg,rgba(35,45,85,0.35),transparent)]",
      )}
    >
      <span
        aria-hidden="true"
        className="flex h-12 w-12 items-center justify-center rounded-full border border-line bg-navy-850 text-peri-300"
      >
        ▶
      </span>
      <p className="text-sm font-medium text-ink-200">
        {unavailable
          ? "The overview video is unavailable right now."
          : "Product overview video"}
      </p>
      <p className="max-w-md text-xs leading-relaxed text-ink-400">
        {unavailable
          ? "The interactive demonstration below covers the same ground, live."
          : "The recorded walkthrough will appear here. In the meantime, the interactive demonstration below runs on the same synthetic portfolio."}
      </p>
      <a
        href="#live-demo"
        className="text-xs font-semibold text-peri-300 underline underline-offset-4 hover:text-peri-200"
      >
        Go to the live demonstration
      </a>
    </Card>
  );
}
