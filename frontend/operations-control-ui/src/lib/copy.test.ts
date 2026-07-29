import { describe, expect, it } from "vitest";
import { copy } from "./copy";

const FORBIDDEN = /blob:|sha256|\.py|\.json|\.yaml|manifest|canonical|registry|traceback/i;

function collectStrings(value: unknown, path: string, out: [string, string][]): void {
  if (typeof value === "string") {
    out.push([path, value]);
  } else if (typeof value === "object" && value !== null) {
    for (const [key, child] of Object.entries(value)) {
      collectStrings(child, `${path}.${key}`, out);
    }
  }
}

describe("copy", () => {
  it("contains no technical terminology in any user-facing string", () => {
    const strings: [string, string][] = [];
    collectStrings(copy, "copy", strings);
    expect(strings.length).toBeGreaterThan(50);
    for (const [path, text] of strings) {
      expect(text, `${path} contains forbidden technical wording`).not.toMatch(FORBIDDEN);
    }
  });
});
