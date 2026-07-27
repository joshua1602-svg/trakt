import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.E2E_PORT ?? 3100);
const baseURL = process.env.E2E_BASE_URL ?? `http://127.0.0.1:${PORT}`;

/**
 * End-to-end suite.
 *
 * Runs against a real production build by default, so what is tested is what
 * ships. `PLAYWRIGHT_CHROMIUM_PATH` lets a sandbox point at a pre-installed
 * browser instead of downloading one.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : [["list"]],
  timeout: 30_000,
  use: {
    baseURL,
    trace: "on-first-retry",
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH }
      : {},
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : {
        command: `npm run build && npx next start --port ${PORT}`,
        url: `${baseURL}/api/health`,
        timeout: 180_000,
        reuseExistingServer: !process.env.CI,
        env: {
          APPLICATION_ENV: "test",
          DEMO_SESSION_SECRET: "e2e-session-secret-at-least-32-characters",
          LEAD_DELIVERY_PROVIDER: "console",
          NEXT_PUBLIC_SITE_URL: baseURL,
        },
      },
});
