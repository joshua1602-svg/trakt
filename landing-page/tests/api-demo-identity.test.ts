import { afterEach, describe, expect, it, vi } from "vitest";

import packJson from "@data/demo-pack.json";
import {
  DEMO_IDENTITY_FIELDS,
  buildDemoIdentity,
  type DemoIdentity,
} from "@/lib/demo-identity";
import { GET } from "@/app/api/demo-identity/route";
import {
  IDENTITY_FIELDS,
  compareIdentity,
  expectedIdentityFromPack,
  markerFromHtml,
  markerFromIdentity,
} from "../scripts/verify-deployed-identity.mjs";
import type { DemoPack } from "@/types/demo";

/**
 * The deployment identity control.
 *
 * `/api/demo-identity` is the contract the post-deploy gate asserts against, so
 * these tests are about the contract as much as the code: the exact field set,
 * exact values from the committed pack, fail-closed behaviour on an incomplete
 * pack, and the absence of anything that should not be public.
 *
 * The last group is the one that earns its keep. The verifier that runs in CI
 * cannot import the TypeScript it is checking, so it carries its own copy of
 * the projection. These tests build the document both ways and compare them —
 * without that, the two could drift apart and the gate would go on passing
 * while checking a field set the endpoint no longer serves.
 */

const pack = packJson as unknown as DemoPack;

afterEach(() => {
  vi.resetModules();
});

describe("GET /api/demo-identity", () => {
  it("serves exactly the identity field set, and nothing else", async () => {
    const response = GET();
    expect(response.status).toBe(200);

    const body = await response.json();
    expect(Object.keys(body).sort()).toEqual([...DEMO_IDENTITY_FIELDS].sort());
  });

  it("serves the committed pack's values, exactly", async () => {
    const body = await GET().json();

    expect(body.clientId).toBe(pack.source.clientId);
    expect(body.portfolioId).toBe(pack.source.portfolioId);
    expect(body.sourceFingerprint).toBe(pack.source.canonicalSha256);
    expect(body.reportingDate).toBe(pack.source.reportingDate);
    expect(body.currency).toBe(pack.source.currency);
    expect(body.totalBalanceDisplay).toBe(pack.portfolio.totalBalanceDisplay);
    expect(body.platformBalanceDisplay).toBe(pack.portfolio.platformBalanceDisplay);
    expect(body.exposures).toBe(pack.source.exposures);
    expect(body.packVersion).toBe(pack.packVersion);
    expect(body.synthetic).toBe(true);
  });

  it("publishes a full SHA-256 of the canonical extract", async () => {
    const body = await GET().json();
    expect(body.sourceFingerprint).toMatch(/^[0-9a-f]{64}$/);
  });

  it("is never cached, so it cannot vouch for a bundle that stopped running", () => {
    expect(GET().headers.get("cache-control")).toBe("no-store");
  });

  it("carries no secret, path, address or internal engine name", async () => {
    const serialised = JSON.stringify(await GET().json());

    for (const forbidden of [
      "mi_agent",
      "mi_service",
      "synthetic_demo",
      "adapters",
      "blob",
      "AccountKey",
      "InstrumentationKey",
      "@",
      "/home/",
      ".csv",
      ".json",
      "http",
    ]) {
      expect(serialised).not.toContain(forbidden);
    }
  });

  it("names only the synthetic client, and says so", async () => {
    const body = await GET().json();
    expect(body.clientId).toContain("demo");
    expect(body.synthetic).toBe(true);
  });
});

describe("fail-closed behaviour", () => {
  const complete = buildDemoIdentity() as DemoIdentity;

  it("builds a complete document from the committed pack", () => {
    expect(complete).not.toBeNull();
  });

  it("refuses a pack with no source identity block", () => {
    const broken = { ...pack, source: undefined } as unknown as DemoPack;
    expect(buildDemoIdentity(broken)).toBeNull();
  });

  it("refuses a pack whose fingerprint is missing or malformed", () => {
    for (const bad of ["", "   ", "not-a-digest", "ABC123", "a".repeat(63)]) {
      const broken = {
        ...pack,
        source: { ...pack.source, canonicalSha256: bad },
      } as DemoPack;
      expect(buildDemoIdentity(broken)).toBeNull();
    }
  });

  it("refuses a pack with a blank client or portfolio id", () => {
    expect(
      buildDemoIdentity({ ...pack, source: { ...pack.source, clientId: "" } } as DemoPack),
    ).toBeNull();
    expect(
      buildDemoIdentity({
        ...pack,
        source: { ...pack.source, portfolioId: "  " },
      } as DemoPack),
    ).toBeNull();
  });

  it("refuses a pack with no displayed total", () => {
    const broken = {
      ...pack,
      portfolio: { ...pack.portfolio, totalBalanceDisplay: "" },
    } as DemoPack;
    expect(buildDemoIdentity(broken)).toBeNull();
  });

  it("returns 503 with no partial fields when the identity cannot be built", async () => {
    vi.resetModules();
    vi.doMock("@/lib/demo-identity", async (importOriginal) => ({
      ...(await importOriginal<typeof import("@/lib/demo-identity")>()),
      buildDemoIdentity: () => null,
    }));

    const route = await import("@/app/api/demo-identity/route");
    const response = route.GET();
    expect(response.status).toBe(503);

    const body = await response.json();
    expect(body).toEqual({ status: "unavailable" });
    // Not one field of a half-built document, which would otherwise satisfy a
    // comparison of whatever survived.
    for (const field of DEMO_IDENTITY_FIELDS) {
      expect(body).not.toHaveProperty(field);
    }

    vi.doUnmock("@/lib/demo-identity");
  });
});

describe("the deploy verifier agrees with the endpoint", () => {
  it("checks the same field set the endpoint serves", () => {
    expect([...IDENTITY_FIELDS].sort()).toEqual([...DEMO_IDENTITY_FIELDS].sort());
  });

  it("derives the same document from the same pack", () => {
    expect(expectedIdentityFromPack(packJson)).toEqual({ ...buildDemoIdentity() });
  });

  it("passes when the served document matches", async () => {
    const expected = expectedIdentityFromPack(packJson);
    const served = await GET().json();
    expect(compareIdentity(expected, served)).toEqual({ ok: true, problems: [] });
  });
});

describe("the deploy verifier fails closed", () => {
  const expected = expectedIdentityFromPack(packJson);
  const good = () => ({ ...expected });

  it("fails when the client id differs", () => {
    const result = compareIdentity(expected, { ...good(), clientId: "other_client" });
    expect(result.ok).toBe(false);
    expect(result.problems.join(" ")).toContain("clientId");
  });

  it("fails when the portfolio id differs", () => {
    const result = compareIdentity(expected, { ...good(), portfolioId: "ALP_Platform_202603" });
    expect(result.ok).toBe(false);
    expect(result.problems.join(" ")).toContain("portfolioId");
  });

  it("fails when the source fingerprint differs", () => {
    const result = compareIdentity(expected, { ...good(), sourceFingerprint: "0".repeat(64) });
    expect(result.ok).toBe(false);
    expect(result.problems.join(" ")).toContain("sourceFingerprint");
  });

  it("fails when the total balance differs", () => {
    const result = compareIdentity(expected, {
      ...good(),
      totalBalanceDisplay: "£37,270,062",
    });
    expect(result.ok).toBe(false);
    expect(result.problems.join(" ")).toContain("totalBalanceDisplay");
  });

  it("fails when a field is missing", () => {
    const served = good();
    delete (served as Record<string, unknown>).sourceFingerprint;
    const result = compareIdentity(expected, served);
    expect(result.ok).toBe(false);
    expect(result.problems.join(" ")).toContain("missing field");
  });

  it("fails when an unreviewed field appears", () => {
    const result = compareIdentity(expected, { ...good(), leadEmail: "someone@example.com" });
    expect(result.ok).toBe(false);
    expect(result.problems.join(" ")).toContain("unexpected field");
  });

  it("fails when a value's type has drifted", () => {
    const result = compareIdentity(expected, { ...good(), exposures: "118" });
    expect(result.ok).toBe(false);
    expect(result.problems.join(" ")).toContain("type string");
  });

  it("fails on a malformed document", () => {
    for (const malformed of [null, "ok", 42, [], undefined]) {
      expect(compareIdentity(expected, malformed).ok).toBe(false);
    }
  });

  it("reports every difference at once, not the first", () => {
    const result = compareIdentity(expected, {
      ...good(),
      clientId: "x",
      portfolioId: "y",
      sourceFingerprint: "f".repeat(64),
    });
    expect(result.problems.length).toBeGreaterThanOrEqual(3);
  });
});

describe("the homepage marker corroborates the endpoint", () => {
  it("reads the marker Next renders", () => {
    const html =
      '<head><meta name="viewport" content="width=device-width"/>' +
      '<meta name="trakt:pack" content="alderbridge_demo/ALP_Platform_202606/2026-06-30"/></head>';
    expect(markerFromHtml(html)).toBe("alderbridge_demo/ALP_Platform_202606/2026-06-30");
  });

  it("returns null when the marker is absent, rather than guessing", () => {
    expect(markerFromHtml("<head><title>Trakt</title></head>")).toBeNull();
    expect(markerFromHtml("")).toBeNull();
    expect(markerFromHtml(null)).toBeNull();
  });

  it("derives the same marker the layout publishes", () => {
    const identity = buildDemoIdentity() as DemoIdentity;
    expect(markerFromIdentity(identity)).toBe(
      [pack.source.clientId, pack.source.portfolioId, pack.source.reportingDate].join("/"),
    );
  });
});
