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
