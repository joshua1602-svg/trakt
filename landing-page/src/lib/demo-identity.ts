import "server-only";

import { demoPack } from "@/lib/demo-pack";
import type { DemoPack } from "@/types/demo";

/**
 * The identity of the demo pack this deployment is actually serving.
 *
 * Published so that a deploy can be *verified* rather than assumed. Everything
 * here is a direct projection of `data/demo-pack.json` — no derivation, no
 * formatting, no runtime state — so a served value that differs from the
 * committed pack can only mean the running bundle was built from a different
 * pack.
 *
 * Every field is synthetic and already public: the client and portfolio ids are
 * of a fabricated lender, the balances are the figures the page itself prints,
 * and the fingerprint is a SHA-256 of a synthetic canonical extract that is
 * committed to this repository. Nothing here is a secret, and nothing here is
 * derived from a real portfolio. That is a constraint on what may ever be added
 * to this object, not just a description of what is in it today.
 */
export interface DemoIdentity {
  /** Fabricated client the synthetic dataset belongs to. */
  clientId: string;
  /** Governed portfolio id — the single portfolio this demo will answer for. */
  portfolioId: string;
  /** SHA-256 of the canonical extract the pack was generated from. */
  sourceFingerprint: string;
  /** Reporting date of the current period. */
  reportingDate: string;
  currency: string;
  /** Sponsor scope — including the sold, derecognised book. */
  totalBalanceDisplay: string;
  /** Platform scope — warehoused only. */
  platformBalanceDisplay: string;
  /** Exposure count behind the sponsor total. */
  exposures: number;
  packVersion: number;
  /** Always true. The demonstration has never run on client data. */
  synthetic: true;
}

/**
 * The complete field set of the identity document, in response order.
 *
 * The deploy gate compares the served document against this exact key set, so
 * adding a field here is a deliberate change to a control surface and must be
 * made in `scripts/verify-deployed-identity.mjs` at the same time. The pair is
 * held together by `tests/api-demo-identity.test.ts`, which fails if they drift.
 */
export const DEMO_IDENTITY_FIELDS = [
  "clientId",
  "portfolioId",
  "sourceFingerprint",
  "reportingDate",
  "currency",
  "totalBalanceDisplay",
  "platformBalanceDisplay",
  "exposures",
  "packVersion",
  "synthetic",
] as const;

/** A 64-character lowercase hex digest, and nothing else. */
const SHA256_HEX = /^[0-9a-f]{64}$/;

/**
 * Project the pack onto the identity document, or `null` if it cannot be built
 * completely.
 *
 * Partial is not an option. A document missing its fingerprint would still
 * satisfy a naive comparison of the fields that *were* present, so the
 * endpoint fails closed instead: no complete identity, no 200.
 */
export function buildDemoIdentity(pack: DemoPack = demoPack): DemoIdentity | null {
  const source = pack?.source;
  const portfolio = pack?.portfolio;
  if (!source || !portfolio) return null;

  const identity: DemoIdentity = {
    clientId: source.clientId,
    portfolioId: source.portfolioId,
    sourceFingerprint: source.canonicalSha256,
    reportingDate: source.reportingDate,
    currency: source.currency,
    totalBalanceDisplay: portfolio.totalBalanceDisplay,
    platformBalanceDisplay: portfolio.platformBalanceDisplay,
    exposures: source.exposures,
    packVersion: pack.packVersion,
    synthetic: true,
  };

  for (const field of DEMO_IDENTITY_FIELDS) {
    const value = identity[field];
    if (typeof value === "string" && value.trim() === "") return null;
    if (typeof value === "number" && !Number.isFinite(value)) return null;
    if (value === undefined || value === null) return null;
  }

  if (!SHA256_HEX.test(identity.sourceFingerprint)) return null;
  if (identity.exposures <= 0 || identity.packVersion <= 0) return null;

  return identity;
}
