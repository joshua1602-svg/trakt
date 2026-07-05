import { describe, it, expect } from "vitest";
import { resolveAgentClientConfig } from "./index";

describe("resolveAgentClientConfig", () => {
  it("uses the HTTP client when a URL is set (dev)", () => {
    const cfg = resolveAgentClientConfig({ VITE_AGENT_API_URL: "http://api", PROD: false });
    expect(cfg.isMock).toBe(false);
    expect(cfg.misconfigured).toBe(false);
    expect(cfg.url).toBe("http://api");
  });

  it("uses the HTTP client when a URL is set (prod)", () => {
    const cfg = resolveAgentClientConfig({ VITE_AGENT_API_URL: "http://api", PROD: true });
    expect(cfg.isMock).toBe(false);
    expect(cfg.misconfigured).toBe(false);
  });

  it("falls back to the mock in dev without flagging misconfiguration", () => {
    const cfg = resolveAgentClientConfig({ PROD: false });
    expect(cfg.isMock).toBe(true);
    expect(cfg.misconfigured).toBe(false);
  });

  it("FLAGS misconfiguration when a production build has no URL and no explicit mock", () => {
    const cfg = resolveAgentClientConfig({ PROD: true });
    expect(cfg.isMock).toBe(true);
    expect(cfg.misconfigured).toBe(true);
  });

  it("does NOT flag misconfiguration when the mock is an explicit opt-in, even in prod", () => {
    const cfg = resolveAgentClientConfig({ VITE_AGENT_MODE: "mock", PROD: true });
    expect(cfg.isMock).toBe(true);
    expect(cfg.explicitMock).toBe(true);
    expect(cfg.misconfigured).toBe(false);
  });

  it("explicit mock overrides a configured URL", () => {
    const cfg = resolveAgentClientConfig({
      VITE_AGENT_API_URL: "http://api",
      VITE_AGENT_MODE: "mock",
      PROD: true,
    });
    expect(cfg.isMock).toBe(true);
    expect(cfg.misconfigured).toBe(false);
  });
});
