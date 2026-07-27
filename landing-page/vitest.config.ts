import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@data": path.resolve(__dirname, "./data"),
      // `server-only` throws when imported outside a React Server Component
      // build. The modules that import it are plain Node under test.
      "server-only": path.resolve(__dirname, "./tests/stubs/server-only.ts"),
    },
  },
  test: {
    // Absolute, so the suite behaves the same whichever directory it is run
    // from (the repository root has its own tooling).
    root: __dirname,
    environment: "jsdom",
    globals: true,
    setupFiles: [path.resolve(__dirname, "./tests/setup.ts")],
    include: ["tests/**/*.test.{ts,tsx}"],
    restoreMocks: true,
  },
});
