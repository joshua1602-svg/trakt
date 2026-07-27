import type { ReactNode } from "react";

/** Small shared primitives. Deliberately plain — no runtime style engine. */

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Section({
  id,
  children,
  className,
  label,
}: {
  id: string;
  children: ReactNode;
  className?: string;
  label?: string;
}) {
  return (
    <section
      id={id}
      aria-labelledby={label ? `${id}-heading` : undefined}
      className={cx("scroll-mt-24 px-5 sm:px-8", className)}
    >
      <div className="mx-auto w-full max-w-6xl">{children}</div>
    </section>
  );
}

export function SectionHeading({
  id,
  eyebrow,
  title,
  intro,
  align = "left",
}: {
  id: string;
  eyebrow?: string;
  title: string;
  intro?: string;
  align?: "left" | "center";
}) {
  return (
    <header className={cx("max-w-3xl", align === "center" && "mx-auto text-center")}>
      {eyebrow ? (
        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.16em] text-peri-400">
          {eyebrow}
        </p>
      ) : null}
      <h2
        id={`${id}-heading`}
        className="text-balance text-2xl font-semibold leading-tight tracking-tight text-ink-100 sm:text-3xl"
      >
        {title}
      </h2>
      {intro ? (
        <p className="mt-4 text-[15px] leading-relaxed text-ink-300">{intro}</p>
      ) : null}
    </header>
  );
}

export function Card({
  children,
  className,
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "li" | "article";
}) {
  return (
    <Tag
      className={cx(
        "rounded-xl border border-line bg-navy-900/70 p-5 sm:p-6",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "synthetic" | "positive";
}) {
  const tones = {
    neutral: "border-line bg-navy-850 text-ink-300",
    synthetic: "border-amber-400/35 bg-amber-400/10 text-amber-400",
    positive: "border-mint-400/35 bg-mint-400/10 text-mint-400",
  } as const;
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium tracking-wide",
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

const buttonBase =
  "inline-flex items-center justify-center gap-2 rounded-lg px-5 py-3 text-sm font-semibold " +
  "transition-colors disabled:cursor-not-allowed disabled:opacity-55";

export const buttonStyles = {
  primary: cx(buttonBase, "bg-peri-400 text-navy-950 hover:bg-peri-300"),
  secondary: cx(
    buttonBase,
    "border border-line bg-navy-850 text-ink-100 hover:border-peri-500 hover:bg-navy-800",
  ),
  ghost: cx(buttonBase, "text-ink-300 hover:text-ink-100"),
} as const;
