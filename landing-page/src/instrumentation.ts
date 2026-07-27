/**
 * Next.js startup hook — the one place production configuration is enforced.
 *
 * Runs once per server process, before any request is served. An unsafe
 * production configuration throws here, so the deployment fails visibly at
 * boot instead of quietly serving a site that drops leads, issues forgeable
 * sessions or advertises a localhost canonical URL.
 *
 * Outside production this only reports, so local development is never blocked
 * by the absence of a mail provider or a shared counter store.
 */
export async function register(): Promise<void> {
  // Only the Node.js server runtime; the edge runtime has no config to check.
  if (process.env.NEXT_RUNTIME !== "nodejs") return;

  const { assertConfig, validateConfig } = await import("@/lib/config");
  const { assertRateLimitStore } = await import("@/lib/rate-limit");
  const { IS_PRODUCTION } = await import("@/lib/env");

  if (!IS_PRODUCTION) {
    const issues = validateConfig();
    if (issues.length > 0) {
      console.warn(
        "[config] non-production configuration notes:\n" +
          issues.map((i) => `  [${i.code}] ${i.message}`).join("\n"),
      );
    }
    return;
  }

  assertRateLimitStore(true);
  assertConfig({ production: true });
  console.info("[config] production configuration validated");
}
