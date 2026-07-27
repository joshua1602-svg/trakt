import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ANALYTICS_EVENTS,
  MAX_ANALYTICS_BODY_BYTES,
  isAnalyticsEvent,
  sanitiseProperties,
} from "@/lib/analytics-events";
import {
  MAX_ATTRIBUTION_LENGTH,
  attributionFromSearch,
  sanitiseAttribution,
} from "@/lib/attribution";
import { resetRateLimits } from "@/lib/rate-limit";

/**
 * The collector is loaded under `firstparty`, because the default provider
 * (`none`) accepts and discards by design — there would be nothing to assert.
 */
async function loadCollector() {
  vi.resetModules();
  process.env.NEXT_PUBLIC_ANALYTICS_PROVIDER = "firstparty";
  const mod = await import("@/app/api/analytics/route");
  return mod.POST;
}

let analyticsPost: (request: Request) => Promise<Response>;

let ipCounter = 0;
function event(body: unknown, headers: Record<string, string> = {}) {
  ipCounter += 1;
  return new Request("http://localhost:3000/api/analytics", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      host: "localhost:3000",
      origin: "http://localhost:3000",
      "x-forwarded-for": `192.0.2.${ipCounter % 250}`,
      ...headers,
    },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

/** Captures what the collector actually wrote. */
function captureLogs(): string[] {
  const lines: string[] = [];
  vi.spyOn(console, "info").mockImplementation((...args) => {
    lines.push(args.map(String).join(" "));
  });
  return lines;
}

beforeEach(async () => {
  resetRateLimits();
  analyticsPost = await loadCollector();
});

afterEach(() => {
  delete process.env.NEXT_PUBLIC_ANALYTICS_PROVIDER;
  vi.resetModules();
});

describe("the event contract", () => {
  it("accepts exactly the twelve allow-listed events", () => {
    expect([...ANALYTICS_EVENTS]).toEqual([
      "hero_demo_click",
      "video_play",
      "demo_open",
      "suggested_question_click",
      "typed_question_submit",
      "demo_answer_returned",
      "demo_refusal_returned",
      "report_preview_opened",
      "capability_interaction",
      "book_demo_click",
      "lead_submit_success",
      "lead_submit_failure",
    ]);
  });

  it.each(["", "arbitrary_event", "DROP TABLE", "hero_demo_click ", 42, null, {}])(
    "rejects %j as an event name",
    (value) => {
      expect(isAnalyticsEvent(value)).toBe(false);
    },
  );
});

describe("property sanitisation", () => {
  it("keeps only allow-listed keys", () => {
    const result = sanitiseProperties({
      intentId: "region_exposure",
      question: "what is the balance",
      answer: "£5.4MM",
      email: "someone@example.com",
    });
    expect(result).toEqual({ intentId: "region_exposure" });
  });

  it("strips characters that free text would need", () => {
    expect(sanitiseProperties({ intentId: "a b<script>c" })).toEqual({ intentId: "abscriptc" });
  });

  it("caps the length of a value", () => {
    const result = sanitiseProperties({ intentId: "x".repeat(500) });
    expect(result.intentId?.length).toBe(64);
  });
});

describe("POST /api/analytics", () => {
  it("records an allow-listed event", async () => {
    const lines = captureLogs();
    const response = await analyticsPost(
      event({ event: "demo_answer_returned", properties: { intentId: "ltv_band" } }),
    );
    expect(response.status).toBe(204);
    expect(lines.join(" ")).toContain("demo_answer_returned");
    expect(lines.join(" ")).toContain("ltv_band");
  });

  it("never records raw question text, even when it is sent", async () => {
    const lines = captureLogs();
    await analyticsPost(
      event({
        event: "typed_question_submit",
        properties: {
          intentId: "wa_ltv",
          question: "what is the balance for Acme Ltd",
          answer: "the funded book stands at",
          email: "someone@example.com",
        },
      }),
    );
    const written = lines.join(" ");
    expect(written).toContain("typed_question_submit");
    expect(written).not.toContain("Acme");
    expect(written).not.toContain("what is the balance");
    expect(written).not.toContain("someone@example.com");
    expect(written).not.toContain("funded book");
  });

  it("drops an unknown event without recording it", async () => {
    const lines = captureLogs();
    const response = await analyticsPost(event({ event: "arbitrary_event", properties: {} }));
    // Always 204 — a rejected event tells an abuser nothing.
    expect(response.status).toBe(204);
    expect(lines.join(" ")).not.toContain("arbitrary_event");
  });

  it("records no IP or user agent", async () => {
    const lines = captureLogs();
    await analyticsPost(
      event({ event: "hero_demo_click" }, { "user-agent": "Mozilla/5.0 SecretBrowser" }),
    );
    const written = lines.join(" ");
    expect(written).not.toContain("192.0.2");
    expect(written).not.toContain("SecretBrowser");
  });

  it("rejects an oversized body", async () => {
    const lines = captureLogs();
    const response = await analyticsPost(
      event({ event: "hero_demo_click", properties: { section: "x".repeat(MAX_ANALYTICS_BODY_BYTES * 2) } }),
    );
    expect(response.status).toBe(204);
    expect(lines.join(" ")).not.toContain("hero_demo_click");
  });

  it("rejects a cross-origin beacon", async () => {
    const lines = captureLogs();
    const response = await analyticsPost(
      event({ event: "hero_demo_click" }, { origin: "https://evil.example" }),
    );
    expect(response.status).toBe(204);
    expect(lines.join(" ")).not.toContain("hero_demo_click");
  });

  it("rate-limits per IP without ever erroring", async () => {
    const ip = "198.51.100.7";
    const send = () =>
      analyticsPost(
        new Request("http://localhost:3000/api/analytics", {
          method: "POST",
          headers: {
            "content-type": "application/json",
            host: "localhost:3000",
            origin: "http://localhost:3000",
            "x-forwarded-for": ip,
          },
          body: JSON.stringify({ event: "hero_demo_click" }),
        }),
      );
    for (let i = 0; i < 130; i += 1) {
      expect((await send()).status).toBe(204);
    }
  });

  it("survives malformed JSON", async () => {
    const response = await analyticsPost(event("{not json"));
    expect(response.status).toBe(204);
  });
});

describe("campaign attribution", () => {
  it("reads the five UTM fields plus the first-party fields", () => {
    const result = attributionFromSearch(
      "?utm_source=linkedin&utm_medium=paid&utm_campaign=q3-erm&utm_content=card-a" +
        "&utm_term=loan+tape&persona=coo&use_case=investor-reporting&source=newsletter",
    );
    expect(result).toEqual({
      utm_source: "linkedin",
      utm_medium: "paid",
      utm_campaign: "q3-erm",
      utm_content: "card-a",
      utm_term: "loan tape",
      persona: "coo",
      use_case: "investor-reporting",
      source: "newsletter",
    });
  });

  it("ignores parameters outside the allow-list", () => {
    const result = attributionFromSearch("?utm_source=ok&evil=payload&email=a@b.co");
    expect(result).toEqual({ utm_source: "ok" });
  });

  it("strips characters that could ride into an email body", () => {
    const result = sanitiseAttribution({
      utm_source: "<script>alert(1)</script>",
      utm_campaign: "line\nbreak\r\ninjection",
    });
    // "/" is a legitimate campaign-code separator and is kept; the markup
    // delimiters that would make it dangerous are not.
    expect(result.utm_source).toBe("script alert 1 /script");
    expect(result.utm_source).not.toMatch(/[<>]/);
    expect(result.utm_campaign).toBe("line break injection");
    expect(result.utm_campaign).not.toMatch(/[\r\n]/);
  });

  it("caps each field", () => {
    const result = sanitiseAttribution({ utm_campaign: "c".repeat(500) });
    expect(result.utm_campaign?.length).toBe(MAX_ATTRIBUTION_LENGTH);
  });

  it("drops non-string values", () => {
    expect(sanitiseAttribution({ utm_source: 42, utm_medium: null, persona: {} })).toEqual({});
  });
});
