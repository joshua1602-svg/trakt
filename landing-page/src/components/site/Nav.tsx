"use client";

import { useEffect, useRef, useState } from "react";

import { TraktWordmark } from "@/components/site/TraktWordmark";
import { buttonStyles, cx } from "@/components/ui";
import { track } from "@/lib/analytics";

const LINKS = [
  { href: "#query-demo", label: "Demo" },
  { href: "#platform", label: "Platform" },
  { href: "#controls", label: "Risk & Controls" },
  { href: "#intelligence", label: "Intelligence" },
  { href: "#governance", label: "Governance" },
] as const;

export function Nav() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  /**
   * Mobile menu keyboard behaviour: Escape closes, Tab is trapped inside while
   * it is open, and focus returns to the toggle on close — so a keyboard user
   * is never dropped somewhere arbitrary in the page.
   */
  useEffect(() => {
    if (!open) return;

    // Move focus into the menu when it opens.
    const focusables = () =>
      Array.from(
        menuRef.current?.querySelectorAll<HTMLElement>("a[href], button:not([disabled])") ?? [],
      );
    focusables()[0]?.focus();

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        toggleRef.current?.focus();
        return;
      }
      if (event.key !== "Tab") return;

      const items = [toggleRef.current, ...focusables()].filter(
        (node): node is HTMLElement => Boolean(node),
      );
      if (items.length === 0) return;

      const first = items[0];
      const last = items[items.length - 1];
      if (!first || !last) return;

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
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
          ref={toggleRef}
          onClick={() =>
            setOpen((value) => {
              // Closing by the toggle keeps focus where the user left it.
              if (value) toggleRef.current?.focus();
              return !value;
            })
          }
          className="rounded-lg border border-line px-3 py-2 text-sm text-ink-200 md:hidden"
        >
          <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
          <span aria-hidden="true">{open ? "✕" : "☰"}</span>
        </button>
      </nav>

      {open ? (
        <div
          id="mobile-nav"
          ref={menuRef}
          className="border-t border-line bg-navy-950/97 md:hidden"
        >
          <ul className="mx-auto max-w-6xl px-5 py-3">
            {LINKS.map((link) => (
              <li key={link.href}>
                <a
                  href={link.href}
                  onClick={() => {
                    setOpen(false);
                    toggleRef.current?.focus();
                  }}
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
