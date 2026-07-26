/**
 * Lead form validation, shared by the browser form and the API route.
 *
 * The client uses it for immediate feedback; the server re-runs it as the
 * authority. Nothing here trusts the client.
 */

export interface LeadInput {
  name?: unknown;
  email?: unknown;
  company?: unknown;
  role?: unknown;
  message?: unknown;
  consent?: unknown;
  /** Honeypot: must be empty. Hidden from real users and from assistive tech. */
  website?: unknown;
  /** Milliseconds the form was on screen before submission. */
  elapsedMs?: unknown;
}

export interface ValidLead {
  name: string;
  email: string;
  company: string;
  role: string;
  message: string;
}

export const LEAD_FIELD_LIMITS = {
  name: 120,
  email: 254,
  company: 160,
  role: 120,
  message: 2000,
} as const;

export type LeadErrors = Partial<Record<keyof ValidLead | "consent", string>>;

/**
 * Pragmatic address check: one @, a dotted domain, no whitespace. Deliberately
 * not an RFC 5322 parser — the address is verified by someone replying to it.
 */
const EMAIL_RE = /^[^\s@]{1,64}@[^\s@.]+(\.[^\s@.]+)+$/;

/** Free-mail domains are accepted; this only blocks obvious throwaways. */
const BLOCKED_EMAIL_DOMAINS = new Set(["example.com", "test.com", "mailinator.com"]);

function clean(value: unknown, max: number): string {
  if (typeof value !== "string") return "";
  return value
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, max);
}

export function validateLead(input: LeadInput): {
  ok: boolean;
  errors: LeadErrors;
  value: ValidLead;
} {
  const value: ValidLead = {
    name: clean(input.name, LEAD_FIELD_LIMITS.name),
    email: clean(input.email, LEAD_FIELD_LIMITS.email).toLowerCase(),
    company: clean(input.company, LEAD_FIELD_LIMITS.company),
    role: clean(input.role, LEAD_FIELD_LIMITS.role),
    message: clean(input.message, LEAD_FIELD_LIMITS.message),
  };

  const errors: LeadErrors = {};

  if (value.name.length < 2) errors.name = "Please enter your name.";
  if (!EMAIL_RE.test(value.email)) {
    errors.email = "Please enter a valid work email address.";
  } else if (BLOCKED_EMAIL_DOMAINS.has(value.email.split("@")[1] ?? "")) {
    errors.email = "Please use your work email address.";
  }
  if (value.company.length < 2) errors.company = "Please enter your company.";
  if (value.role.length < 2) errors.role = "Please enter your role.";
  if (input.consent !== true) {
    errors.consent = "Please confirm you are happy for us to contact you.";
  }

  return { ok: Object.keys(errors).length === 0, errors, value };
}

/** Bot signals, evaluated server-side only. */
export function botSignals(
  input: LeadInput,
  minFillMs: number,
): { bot: boolean; reason: string | null } {
  if (typeof input.website === "string" && input.website.trim() !== "") {
    return { bot: true, reason: "honeypot" };
  }
  const elapsed = typeof input.elapsedMs === "number" ? input.elapsedMs : 0;
  if (minFillMs > 0 && elapsed < minFillMs) {
    return { bot: true, reason: "too_fast" };
  }
  return { bot: false, reason: null };
}
