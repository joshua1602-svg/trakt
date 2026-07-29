import "@testing-library/jest-dom/vitest";

if (typeof window !== "undefined") {
  window.HTMLElement.prototype.scrollTo = window.HTMLElement.prototype.scrollTo || (() => {});
  window.HTMLElement.prototype.scrollIntoView =
    window.HTMLElement.prototype.scrollIntoView || (() => {});
}
