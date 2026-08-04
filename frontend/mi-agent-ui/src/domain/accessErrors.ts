/**
 * Governed access refusals, as the UI needs to tell them apart.
 *
 * A signed-in user who has not been provisioned in Trakt's access directory is
 * not looking at a failure — nothing is broken, nothing is retryable, and the
 * fix is an administrator adding them. Rendering that as "Something went wrong"
 * with a Retry button sends them round a loop that cannot terminate, which is
 * exactly what happened while Static Web Apps invitation roles were the
 * authority.
 *
 * The codes mirror `trakt_core.errors.ErrorCode`; `accessErrors.test.ts` pins
 * the message against the backend constant so the two cannot drift.
 */

/** `ErrorCode.ACCESS_NOT_PROVISIONED` — authenticated, but not in the directory. */
export const ACCESS_NOT_PROVISIONED = "ACCESS_NOT_PROVISIONED";
/** `ErrorCode.ACCESS_DISABLED` — in the directory, switched off. */
export const ACCESS_DISABLED = "ACCESS_DISABLED";
/** `ErrorCode.TENANT_MISMATCH` — provisioned for a different deployment. */
export const TENANT_MISMATCH = "TENANT_MISMATCH";

/** Verbatim from `trakt_core.access.NOT_PROVISIONED_MESSAGE`. */
export const NOT_PROVISIONED_MESSAGE =
  "Your account is signed in but has not been provisioned for Trakt access.";

export interface AccessRefusal {
  readonly title: string;
  readonly detail: string;
  /** Retrying cannot clear any of these, so the UI must not offer it. */
  readonly retryable: false;
}

const REFUSALS: Record<string, AccessRefusal> = {
  [ACCESS_NOT_PROVISIONED]: {
    title: "Access not provisioned",
    detail:
      NOT_PROVISIONED_MESSAGE +
      " Ask your Trakt administrator to add your account.",
    retryable: false,
  },
  [ACCESS_DISABLED]: {
    title: "Access disabled",
    detail:
      "Your Trakt access has been disabled. Contact your Trakt administrator.",
    retryable: false,
  },
  [TENANT_MISMATCH]: {
    title: "Wrong Trakt tenant",
    detail:
      "Your account is provisioned for a different Trakt tenant than this one.",
    retryable: false,
  },
};

/**
 * The refusal an error represents, or `undefined` if it is an ordinary failure.
 *
 * Matches on the code when one survived the call, and otherwise on the message
 * text. The text path matters because callers compose errors into sentences
 * ("The MI Agent API could not be reached — {message}"), so a substring match
 * is the only thing that still works once the message has been wrapped.
 */
export function accessRefusal(
  input: { code?: string; message?: string } | string | null | undefined,
): AccessRefusal | undefined {
  if (!input) return undefined;
  const code = typeof input === "string" ? undefined : input.code;
  const message = typeof input === "string" ? input : (input.message ?? "");

  if (code && REFUSALS[code]) return REFUSALS[code];
  for (const [candidate, refusal] of Object.entries(REFUSALS)) {
    if (message.includes(candidate)) return refusal;
  }
  if (message.includes(NOT_PROVISIONED_MESSAGE)) {
    return REFUSALS[ACCESS_NOT_PROVISIONED];
  }
  return undefined;
}
