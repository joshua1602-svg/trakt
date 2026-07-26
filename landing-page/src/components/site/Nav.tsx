"use client";

import { useEffect, useState } from "react";

import { TraktWordmark } from "@/components/site/TraktWordmark";
import { buttonStyles, cx } from "@/components/ui";
import { track } from "@/lib/analytics";

const LINKS = [
  { href: "#product", label: "Product" },
  { href: "#capabilities", label: "Capabilities" },
  { href: "#how-it-works", label: "How it works" },
  { href: "#governance", label: "Governance" },
] as const;

export function Nav() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Close the mobile menu on Escape, and whenever the viewport grows.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);

    // Older Safari exposes MediaQueryList without addEventListener; the menu
    // still works there, it just does not auto-close on rotation.
    const media = window.matchMedia?.("(min-width: 768px)");
    const onChange = () => media?.matches && setOpen(false);
    media?.addEventListener?.("change", onChange);

    return () => {
      document.removeEventListener("keydown", onKey);
      media?.removeEventListener?.("change", onChange);
    };
  }, [open]);

  return (
    <header
      className={cx(
        "fixed inset-x-0 top-0 z-50 border-b transition-colors",
        scrolled || open
          ? "border-line bg-navy-950/92 backdrop-blur-sm"
          : "border-transparent bg-transparent",
      )}
    >
      <nav aria-label="Primary" className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8">
        <a href="#top" className="flex items-center gap-2.5" aria-label="Trakt — home">
          <TraktWordmark />
        </a>

        <ul className="hidden items-center gap-7 md:flex">
          {LINKS.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className="text-sm text-ink-300 transition-colors hover:text-ink-100"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        <div className="hidden md:block">
          <a
            href="#book-a-demo"
            onClick={() => track("book_demo_click", { source: "nav" })}
            className={cx(buttonStyles.primary, "px-4 py-2 text-[13px]")}
          >
            Book a demo
          </a>
        </div>

        <button
          type="button"
          aria-expanded={open}
          aria-controls="mobile-nav"
          onClick={() => setOpen((value) => !value)}
          className="rounded-lg border border-line px-3 py-2 text-sm text-ink-200 md:hidden"
        >
          <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
          <span aria-hidden="true">{open ? "✕" : "☰"}</span>
        </button>
      </nav>

      {open ? (
        <div id="mobile-nav" className="border-t border-line bg-navy-950/97 md:hidden">
          <ul className="mx-auto max-w-6xl px-5 py-3">
            {LINKS.map((link) => (
              <li key={link.href}>
                <a
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className="block border-b border-line-soft py-3 text-sm text-ink-200"
                >
                  {link.label}
                </a>
              </li>
            ))}
            <li>
              <a
                href="#book-a-demo"
                onClick={() => {
                  track("book_demo_click", { source: "nav_mobile" });
                  setOpen(false);
                }}
                className="block py-3 text-sm font-semibold text-peri-300"
              >
                Book a demo
              </a>
            </li>
          </ul>
        </div>
      ) : null}
    </header>
  );
}
