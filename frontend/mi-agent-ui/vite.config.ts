/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    host: true,
    // Dev proxy so the browser talks to the MI Agent API SAME-ORIGIN (no CORS) —
    // which is what Codespaces / forwarded ports need. With this, run the UI with
    // VITE_AGENT_API_URL=/ and the HTTP client calls the API paths on the dev
    // server, which forwards them (VITE_PROXY_TARGET, default :8000).
    //
    // EVERY path the client calls must be listed. `/me` was previously missing,
    // so the identity request fell through to the dev server and came back as
    // the SPA's index.html (or a 404) instead of reaching the API.
    proxy: Object.fromEntries(
      ["/mi", "/health", "/me", "/v1"].map((route) => [
        route,
        { target: process.env.VITE_PROXY_TARGET || "http://localhost:8000", changeOrigin: true },
      ]),
    ),
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
