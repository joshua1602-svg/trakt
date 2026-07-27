/**
 * Client-safe configuration.
 *
 * Only `NEXT_PUBLIC_*` values appear here, referenced statically so Next inlines
 * them at build time. `lib/env.ts` (server-only, holds secrets) must never be
 * imported from a client component — this module exists so it never needs to be.
 */
export const publicConfig = {
  siteUrl: (process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000").replace(/\/$/, ""),
  demoName: process.env.NEXT_PUBLIC_DEMO_NAME ?? "Synthetic Demo Lender",
  analyticsProvider: (process.env.NEXT_PUBLIC_ANALYTICS_PROVIDER ?? "none") as
    | "none"
    | "appinsights"
    | "firstparty",
  /** Optional hosted URL for the product overview video. */
  videoUrl: process.env.NEXT_PUBLIC_DEMO_VIDEO_URL ?? "",
  videoPoster: process.env.NEXT_PUBLIC_DEMO_VIDEO_POSTER ?? "",
  contactEmail: process.env.NEXT_PUBLIC_CONTACT_EMAIL ?? "",
} as const;
