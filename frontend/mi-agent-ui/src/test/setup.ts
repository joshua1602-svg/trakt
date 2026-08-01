import "@testing-library/jest-dom/vitest";

// Recharts' ResponsiveContainer needs a measurable parent; jsdom reports 0.
// Stub a fixed size so charts render in tests.
if (typeof window !== "undefined") {
  Object.defineProperty(window.HTMLElement.prototype, "offsetWidth", {
    configurable: true,
    value: 800,
  });
  Object.defineProperty(window.HTMLElement.prototype, "offsetHeight", {
    configurable: true,
    value: 400,
  });
  window.ResizeObserver =
    window.ResizeObserver ||
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  // jsdom doesn't implement scroll APIs the chat / artifact panels call.
  window.HTMLElement.prototype.scrollTo = window.HTMLElement.prototype.scrollTo || (() => {});
  window.HTMLElement.prototype.scrollIntoView =
    window.HTMLElement.prototype.scrollIntoView || (() => {});
}

// The shell mirrors navigation (view / sub-tab / scope) into the URL so views
// are shareable. Reset it between tests so one test's navigation doesn't
// deep-link-restore into the next test's fresh render.
import { beforeEach } from "vitest";
beforeEach(() => {
  if (typeof window !== "undefined" && window.history?.replaceState) {
    window.history.replaceState(null, "", window.location.pathname);
  }
});
