"use client";

import { useEffect, useRef, useState } from "react";

import { TraktWordmark } from "@/components/site/TraktWordmark";
import { buttonStyles, cx } from "@/components/ui";
import { track } from "@/lib/analytics";

/**
 * Every navigable section, in page order. This is the mobile menu, where
 * there is room to name the page in full.
 *
 * "Demo" is gone as a label: the page carries more than one, so a nav entry
 * named after the format rather than the section told the reader nothing.
 */
const LINKS = [
  { href: "#query-demo", label: "Portfolio query", desktop: false },
  { href: "#refusal", label: "Boundaries", desktop: false },
  { href: "#platform", label: "Platform", desktop: true },
  { href: "#controls", label: "Risk & controls", desktop: true },
  { href: "#delivery", label: "Delivery", desktop: true },
  { href: "#agents", label: "Agent-to-agent", desktop: true },
  { href: "#governance", label: "Governance", desktop: true },
] as const;

/**
 * The desktop bar takes five of the seven. Measured at 1024 — the narrowest
 * width the bar now appears at — these five come to 525px against a 589px
 * budget once the wordmark and the CTA are subtracted.
 *
 * Portfolio query is dropped because it opens the page and the hero's own
 * primary button already points at it; Boundaries because it is a sub-claim of
 * that demo rather than a destination. Agent-to-agent is in the bar
 * deliberately: it is the section a technical reader comes looking for.
 */
const DESKTOP_LINKS = LINKS.filter((link) => link.desktop);

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
    const media = window.matchMedia?.("(min-width: 1024px)");
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
      <nav aria-label="Primary" className="mx-auto flex h-16 w-full max-w-[1600px] items-center justify-between px-5 sm:px-8 lg:px-12">
        <a href="#top" className="flex items-center gap-2.5" aria-label="Trakt — home">
          <TraktWordmark />
        </a>

        {/* `lg`, not `md`. At 768 the bar's contents came to 701px inside a
            704px row — three pixels of slack, which is not a design and left
            no room for either a longer CTA or a sixth section. Between 768 and
            1023 the menu button is the correct control. */}
        <ul className="hidden items-center gap-7 lg:flex">
          {DESKTOP_LINKS.map((link) => (
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

        {/* Not "Book a demo": the page shows demos of its own, so an invitation
            to book one reads as though those do not count. What is on offer
            here is the same product run against the visitor's portfolio rather
            than the synthetic one, and the label says so. */}
        <div className="hidden lg:block">
          <a
            href="#book-a-demo"
            onClick={() => track("book_demo_click", { source: "nav" })}
            className={cx(buttonStyles.primary, "px-4 py-2 text-[13px]")}
          >
            Demo on your portfolio
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
          className="rounded-lg border border-line px-3 py-2 text-sm text-ink-200 lg:hidden"
        >
          <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
          <span aria-hidden="true">{open ? "✕" : "☰"}</span>
        </button>
      </nav>

      {open ? (
        <div
          id="mobile-nav"
          ref={menuRef}
          className="border-t border-line bg-navy-950/97 lg:hidden"
        >
          <ul className="mx-auto w-full max-w-[1600px] px-5 py-3 sm:px-8 lg:px-12">
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
                Demo on your portfolio
              </a>
            </li>
          </ul>
        </div>
      ) : null}
    </header>
  );
}
