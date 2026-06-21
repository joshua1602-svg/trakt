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
}
