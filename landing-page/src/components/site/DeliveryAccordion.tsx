"use client";

import { useRef, useState } from "react";

import { cx } from "@/components/ui";

/**
 * The delivery model as a horizontal accordion — the page's only one.
 *
 * From 768px up: collapsed panels render as vertical spines with rotated
 * labels; expanding a panel pushes its siblings aside (flex-grow transition,
 * 240ms ease-out — precise, not playful). Panel 1 is open on load and one
 * panel is always open: selecting a spine moves the expansion, it never
 * leaves the row all-collapsed. Arrow keys move between panels; Enter/Space
 * expand (native button semantics). Below 768px the same content is a plain
 * vertical stack with no horizontal behaviour.
 *
 * Panel copy is reused verbatim from the delivery-model claims already in
 * `docs/content-map.md`; the live/roadmap split keeps its meaning — mint is
 * shipped, grey is not.
 */

const PANELS = [
  {
    id: "managed",
    name: "Managed service",
    copy: "Recurring reporting, regulatory output and governance artefacts, produced with no user interaction.",
    status: "available" as const,
  },
  {
    id: "agent",
    name: "Trakt Agent",
    copy: "The full analytical environment: dashboards, charting, drill-through and portfolio investigation.",
    status: "available" as const,
  },
  {
    id: "copilot",
    name: "Copilot",
    copy: "Portfolio questions and artefact requests inside the tools your teams already use.",
    status: "available" as const,
  },
  {
    id: "enterprise-agent",
    name: "Enterprise agent",
    copy: "Trakt running inside a client's own agent estate.",
    status: "roadmap" as const,
  },
  {
    id: "agent-to-agent",
    name: "Agent-to-agent",
    copy: "Upstream and downstream systems consulting the governed layer directly.",
    status: "roadmap" as const,
  },
] as const;

function StatusTag({ status }: { status: (typeof PANELS)[number]["status"] }) {
  return status === "available" ? (
    <span className="text-[11px] font-semibold uppercase tracking-wider text-mint-400">
      Available today
    </span>
  ) : (
    <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-500">
      Roadmap
    </span>
  );
}

export function DeliveryAccordion() {
  const [openIndex, setOpenIndex] = useState(0);
  const spineRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    const current = spineRefs.current.findIndex(
      (node) => node === document.activeElement,
    );
    const base = current === -1 ? openIndex : current;
    const next =
      event.key === "ArrowRight"
        ? Math.min(base + 1, PANELS.length - 1)
        : Math.max(base - 1, 0);
    spineRefs.current[next]?.focus();
  };

  return (
    <>
      {/* ≥768px: the horizontal accordion. */}
      <div
        className="hidden gap-2 md:flex"
        role="group"
        aria-label="Delivery modes"
        onKeyDown={onKeyDown}
      >
        {PANELS.map((panel, index) => {
          const open = index === openIndex;
          return (
            <div
              key={panel.id}
              className={cx(
                "min-w-0 overflow-hidden rounded-xl border transition-[flex-grow] duration-[240ms] ease-out",
                open
                  ? "border-line bg-navy-900/70"
                  : panel.status === "available"
                    ? "border-mint-400/25 bg-navy-900/40"
                    : "border-line-soft bg-navy-900/30",
              )}
              style={{ flexGrow: open ? 6 : 1, flexBasis: 0 }}
            >
              {open ? (
                <div className="flex h-full min-h-[13rem] flex-col p-5">
                  <StatusTag status={panel.status} />
                  <h3
                    className={cx(
                      "mt-2 text-lg font-semibold tracking-tight",
                      panel.status === "available" ? "text-ink-100" : "text-ink-300",
                    )}
                  >
                    {panel.name}
                  </h3>
                  <p className="mt-2 max-w-md text-sm leading-relaxed text-ink-400">
                    {panel.copy}
                  </p>
                  {/* The open panel's own header doubles as the affordance
                      target for keyboard users arriving by arrow key. */}
                  <button
                    ref={(node) => {
                      spineRefs.current[index] = node;
                    }}
                    type="button"
                    aria-expanded="true"
                    aria-label={`${panel.name} — expanded`}
                    className="sr-only"
                  >
                    {panel.name}
                  </button>
                </div>
              ) : (
                <button
                  ref={(node) => {
                    spineRefs.current[index] = node;
                  }}
                  type="button"
                  aria-expanded="false"
                  onClick={() => setOpenIndex(index)}
                  className="flex h-full min-h-[13rem] w-14 flex-col items-center justify-between px-0 py-4 text-left transition-colors hover:bg-navy-850/60"
                >
                  <span
                    aria-hidden="true"
                    className={cx(
                      "h-1.5 w-1.5 rounded-full",
                      panel.status === "available" ? "bg-mint-400" : "bg-ink-500",
                    )}
                  />
                  <span
                    className={cx(
                      "whitespace-nowrap text-[13px] font-medium [writing-mode:vertical-rl]",
                      panel.status === "available" ? "text-ink-200" : "text-ink-500",
                    )}
                  >
                    {panel.name}
                  </span>
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* <768px: a plain vertical stack — no horizontal behaviour. */}
      <ul className="flex flex-col gap-3 md:hidden">
        {PANELS.map((panel) => (
          <li
            key={panel.id}
            className={cx(
              "rounded-xl border p-4",
              panel.status === "available"
                ? "border-mint-400/25 bg-navy-900/60"
                : "border-line-soft bg-navy-900/40",
            )}
          >
            <StatusTag status={panel.status} />
            <h3
              className={cx(
                "mt-1.5 text-[15px] font-semibold",
                panel.status === "available" ? "text-ink-100" : "text-ink-300",
              )}
            >
              {panel.name}
            </h3>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-400">{panel.copy}</p>
          </li>
        ))}
      </ul>
    </>
  );
}
