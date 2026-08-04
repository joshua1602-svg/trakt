import { describe, expect, it } from "vitest";

import {
  ACCESS_DISABLED,
  ACCESS_NOT_PROVISIONED,
  NOT_PROVISIONED_MESSAGE,
  TENANT_MISMATCH,
  accessRefusal,
} from "./accessErrors";

describe("accessRefusal", () => {
  it("recognises an unprovisioned account by code", () => {
    const refusal = accessRefusal({ code: ACCESS_NOT_PROVISIONED });
    expect(refusal?.title).toBe("Access not provisioned");
    expect(refusal?.retryable).toBe(false);
  });

  it("recognises a disabled account and a tenant mismatch", () => {
    expect(accessRefusal({ code: ACCESS_DISABLED })?.title).toBe("Access disabled");
    expect(accessRefusal({ code: TENANT_MISMATCH })?.title).toBe("Wrong Trakt tenant");
  });

  it("still recognises the refusal once the message has been wrapped", () => {
    // AppShell composes errors into a sentence, so by the time the string
    // reaches the error panel the raw message is a substring of a longer one.
    const wrapped = `The MI Agent API could not be reached — ${NOT_PROVISIONED_MESSAGE}`;
    expect(accessRefusal(wrapped)?.title).toBe("Access not provisioned");
  });

  it("recognises the refusal from a code embedded in a message", () => {
    expect(accessRefusal(`something ${ACCESS_DISABLED} happened`)?.title).toBe(
      "Access disabled",
    );
  });

  it("leaves an ordinary failure alone", () => {
    expect(accessRefusal("MI Agent API returned 500 for /mi/snapshots")).toBeUndefined();
    expect(accessRefusal("")).toBeUndefined();
    expect(accessRefusal(null)).toBeUndefined();
    expect(accessRefusal(undefined)).toBeUndefined();
  });

  it("does not treat a 403 on its own as a provisioning problem", () => {
    // A bare 403 with no governed envelope could be the platform refusing, not
    // the directory. Claiming "not provisioned" there would misdiagnose it.
    expect(accessRefusal("MI Agent API returned 403 for /mi/snapshots")).toBeUndefined();
  });
});

// The cross-language pin — that these constants still match trakt_core — lives
// in mi_agent_api/tests/test_governed_access.py::TestFrontendContract. Python
// can read both files without pulling node type declarations into the browser
// build, and the assertion is the same either way.
