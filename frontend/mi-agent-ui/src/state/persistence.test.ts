import { afterEach, describe, expect, it } from "vitest";
import { readUrlContext } from "./persistence";

/**
 * readUrlContext parses the Workspace-restoring params off the URL that a
 * Copilot "Open in Workspace" deep link carries. Absent params must yield an
 * empty object so the existing localStorage flow is untouched.
 */
function setSearch(search: string) {
  window.history.replaceState({}, "", `/${search}`);
}

describe("readUrlContext", () => {
  afterEach(() => setSearch(""));

  it("returns an empty object when no params are present", () => {
    setSearch("");
    expect(readUrlContext()).toEqual({});
  });

  it("parses client, run, question and filters", () => {
    setSearch(
      "?client=client_001&run=mi_2025_10&q=" +
        encodeURIComponent("LTV by region") +
        "&filters=" +
        encodeURIComponent('{"region":"North"}'),
    );
    expect(readUrlContext()).toEqual({
      clientId: "client_001",
      runId: "mi_2025_10",
      question: "LTV by region",
      filters: '{"region":"North"}',
    });
  });

  it("ignores unrelated params and omits absent ones", () => {
    setSearch("?client=ERE&utm_source=copilot");
    expect(readUrlContext()).toEqual({ clientId: "ERE" });
  });
});
