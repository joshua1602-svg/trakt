/**
 * The bearer token has to reach the wire, on every kind of MI API request —
 * GET, the two POSTs, and the deck download — and it must be absent when the
 * feature flag is off. These assert the actual `fetch` calls rather than the
 * helper, because the helper being right is not the same as every call site
 * using it.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentRequest } from "@/domain";
import { HttpAgentClient } from "./index";
import { setAccessTokenProvider } from "@/auth/tokenProvider";

const BASE = "https://trakt-mi-api.azurewebsites.net";

const request: AgentRequest = {
  question: "Show balance by region",
  portfolio: { id: "erm-uk-master", name: "ERM UK — Master", entity: "Trakt SPV I" },
  reporting: { asOf: "2026-05-31" },
};

function jsonResponse(body: unknown) {
  return { ok: true, status: 200, json: async () => body } as unknown as Response;
}

/** The Authorization header of the Nth fetch call, whatever init shape was used. */
function authOf(call: number): string | undefined {
  const init = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls[call][1] as
    RequestInit | undefined;
  const headers = (init?.headers ?? {}) as Record<string, string>;
  return headers.Authorization;
}

afterEach(() => {
  setAccessTokenProvider(null);
  vi.restoreAllMocks();
});

describe("with the bearer path on", () => {
  it("attaches the token to a GET", async () => {
    setAccessTokenProvider(async () => "access-token-1");
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ snapshots: [] })));

    await new HttpAgentClient(BASE).getSnapshots();

    expect(authOf(0)).toBe("Bearer access-token-1");
  });

  it("attaches the token to /me", async () => {
    setAccessTokenProvider(async () => "access-token-me");
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ authenticated: true })));

    await new HttpAgentClient(BASE).getMe();

    expect(authOf(0)).toBe("Bearer access-token-me");
  });

  it("attaches the token to the MI query POST without losing Content-Type", async () => {
    setAccessTokenProvider(async () => "access-token-2");
    vi.stubGlobal("fetch", vi.fn(async () =>
      jsonResponse({ ok: true, answer: "…", artifacts: [], metadata: {} })));

    await new HttpAgentClient(BASE).ask(request);

    const init = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } })
      .mock.calls[0][1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer access-token-2");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(init.method).toBe("POST");
  });

  it("attaches the token to the deck download, which a link could not carry", async () => {
    setAccessTokenProvider(async () => "access-token-3");
    const blob = new Blob(["deck"]);
    vi.stubGlobal("fetch", vi.fn(async () =>
      ({ ok: true, status: 200, blob: async () => blob }) as unknown as Response));

    const out = await new HttpAgentClient(BASE).downloadDeck("client_001", "2026-05");

    expect(out).toBe(blob);
    expect(authOf(0)).toBe("Bearer access-token-3");
    const url = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls[0][0];
    expect(url).toContain("/mi/decks/download?portfolioId=client_001");
  });

  it("sends the request unauthenticated when the user is signed out", async () => {
    setAccessTokenProvider(async () => null);
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ snapshots: [] })));

    await new HttpAgentClient(BASE).getSnapshots();

    expect(authOf(0)).toBeUndefined();
  });

  it("sends the request unauthenticated when acquisition fails, rather than throwing", async () => {
    setAccessTokenProvider(async () => { throw new Error("interaction_required"); });
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ snapshots: [] })));

    await expect(new HttpAgentClient(BASE).getSnapshots()).resolves.toBeTruthy();
    expect(authOf(0)).toBeUndefined();
  });

  it("surfaces a 401 as a normal API error, not a crash", async () => {
    setAccessTokenProvider(async () => "expired");
    vi.stubGlobal("fetch", vi.fn(async () =>
      ({ ok: false, status: 401, statusText: "Unauthorized" }) as unknown as Response));

    await expect(new HttpAgentClient(BASE).getSnapshots()).rejects.toThrow(/401/);
  });
});

describe("with the bearer path off (no provider registered)", () => {
  it("sends no Authorization header at all — the SWA path is untouched", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ snapshots: [] })));

    await new HttpAgentClient("/api").getSnapshots();

    expect(authOf(0)).toBeUndefined();
  });

  it("still sends Content-Type on a POST", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      jsonResponse({ ok: true, answer: "…", artifacts: [], metadata: {} })));

    await new HttpAgentClient("/api").ask(request);

    const init = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } })
      .mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
    expect((init.headers as Record<string, string>).Authorization).toBeUndefined();
  });
});
