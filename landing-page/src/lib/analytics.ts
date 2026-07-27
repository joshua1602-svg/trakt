"use client";

import {
  type AnalyticsEvent,
  type AnalyticsProperties,
  MAX_ANALYTICS_BODY_BYTES,
} from "@/lib/analytics-events";
import {
  ATTRIBUTION_STORAGE_KEY,
  type Attribution,
  attributionFromSearch,
  hasAttribution,
  sanitiseAttribution,
} from "@/lib/attribution";
import { publicConfig } from "@/lib/public-config";

/**
 * Privacy-conscious analytics.
 *
 * Captured: an event name from a fixed allow-list, plus a small set of
 * low-cardinality identifiers. Never captured: the text a visitor types into
 * the demo (intent ids are recorded instead), answer content, lead-form
 * contents, any identifier that persists across sessions, and any third-party
 * cookie. No third-party tracker is loaded by default.
 *
 * The adapter is chosen by `NEXT_PUBLIC_ANALYTICS_PROVIDER`; the default
 * (`none`) is a no-op that sends nothing at all.
 */

export type { AnalyticsEvent, AnalyticsProperties };

interface Adapter {
  track(event: AnalyticsEvent, properties?: AnalyticsProperties): void;
}

/** Development / default. Records nothing, anywhere. */
const noopAdapter: Adapter = {
  track() {
    /* intentionally empty */
  },
};

/**
 * Azure Application Insights via the browser snippet, when — and only when —
 * an operator has loaded it. This module never injects a script tag, so
 * enabling it is an explicit deployment decision.
 */
const appInsightsAdapter: Adapter = {
  track(event, properties) {
    const insights = (
      globalThis as unknown as {
        appInsights?: { trackEvent?: (item: unknown) => void };
      }
    ).appInsights;
    if (insights?.trackEvent) {
      insights.trackEvent({ name: `trakt.landing.${event}`, properties: properties ?? {} });
      return;
    }
    // No snippet present: fall back to the first-party collector so events are
    // not silently lost when App Insights is configured server-side only.
    firstPartyAdapter.track(event, properties);
  },
};

/** First-party collector on this origin. Fire-and-forget, never blocking. */
const firstPartyAdapter: Adapter = {
  track(event, properties) {
    try {
      const body = JSON.stringify({ event, properties: properties ?? {} });
      if (body.length > MAX_ANALYTICS_BODY_BYTES) return;

      if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
        const blob = new Blob([body], { type: "application/json" });
        if (navigator.sendBeacon("/api/analytics", blob)) return;
        // sendBeacon refuses when the queue is full; fall through to fetch.
      }

      void fetch("/api/analytics", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        keepalive: true,
      }).catch(() => undefined);
    } catch {
      /* analytics must never break the page */
    }
  },
};

function adapter(): Adapter {
  switch (publicConfig.analyticsProvider) {
    case "appinsights":
      return appInsightsAdapter;
    case "firstparty":
      return firstPartyAdapter;
    default:
      return noopAdapter;
  }
}

export function track(event: AnalyticsEvent, properties?: AnalyticsProperties): void {
  if (typeof window === "undefined") return;
  try {
    adapter().track(event, properties);
  } catch {
    /* never let instrumentation surface to the visitor */
  }
}

/* -------------------------------------------------------------------------- */
/* Campaign attribution                                                       */
/* -------------------------------------------------------------------------- */

/**
 * Capture campaign parameters from the landing URL into `sessionStorage`.
 *
 * Session storage, not a cookie and not `localStorage`: attribution should last
 * exactly as long as the visit that carried it, and should not be readable on a
 * later, unrelated one. First write wins, so a mid-visit navigation cannot
 * overwrite the campaign the visitor actually arrived on.
 */
export function captureAttribution(): void {
  if (typeof window === "undefined") return;
  try {
    const existing = window.sessionStorage.getItem(ATTRIBUTION_STORAGE_KEY);
    if (existing) return;

    const fromQuery = attributionFromSearch(window.location.search);
    if (!hasAttribution(fromQuery) && document.referrer) {
      try {
        const referrer = new URL(document.referrer);
        if (referrer.host !== window.location.host) {
          fromQuery.source = referrer.hostname.slice(0, 64);
        }
      } catch {
        /* an unparseable referrer is simply not recorded */
      }
    }

    if (!hasAttribution(fromQuery)) return;
    window.sessionStorage.setItem(ATTRIBUTION_STORAGE_KEY, JSON.stringify(fromQuery));
  } catch {
    /* storage disabled (private mode, or a strict policy) — carry on without */
  }
}

/** Read the captured attribution, to attach to a lead submission. */
export function readAttribution(): Attribution {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.sessionStorage.getItem(ATTRIBUTION_STORAGE_KEY);
    if (!raw) return {};
    return sanitiseAttribution(JSON.parse(raw));
  } catch {
    return {};
  }
}
