/**
 * Types for the deploy-time identity verifier.
 *
 * The verifier is plain ESM because the deploy job runs it with bare `node`,
 * with no build step and no dev dependencies installed. These declarations
 * exist so the test suite can import its pure helpers under `strict` and hold
 * them against the endpoint they check.
 */

export declare const IDENTITY_FIELDS: string[];

export declare function expectedIdentityFromPack(pack: unknown): Record<string, unknown>;

export declare function compareIdentity(
  expected: Record<string, unknown>,
  served: unknown,
): { ok: boolean; problems: string[] };

export declare function markerFromHtml(html: string | null | undefined): string | null;

export declare function markerFromIdentity(identity: {
  clientId: string;
  portfolioId: string;
  reportingDate: string;
}): string;
