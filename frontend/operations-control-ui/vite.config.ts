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
    port: 5174,
    host: true,
    // Dev proxy so the browser talks to the Operations Control API same-origin
    // (no CORS). Run the API locally and the dev server forwards these paths.
    proxy: Object.fromEntries(
      ["/ops", "/health"].map((route) => [
        route,
        {
          target: process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8100",
          changeOrigin: true,
        },
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
