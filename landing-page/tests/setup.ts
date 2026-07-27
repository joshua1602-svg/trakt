import "@testing-library/jest-dom/vitest";

// The API routes read these at import time. Fixed values keep the suite
// deterministic and independent of the developer's shell.
process.env.APPLICATION_ENV = "test";
process.env.DEMO_SESSION_SECRET = "test-secret-value-at-least-32-chars-long";
process.env.LEAD_DELIVERY_PROVIDER = "console";
