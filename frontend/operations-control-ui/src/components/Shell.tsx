import type { ReactNode } from "react";
import { NavLink, Link } from "react-router-dom";
import {
  Home,
  ListChecks,
  GitBranch,
  BookOpen,
  Clock,
  Building2,
  FlaskConical,
  Plus,
  SlidersHorizontal,
 Gauge } from "lucide-react";
import clsx from "clsx";
import { useSession } from "@/api/session";
import { copy } from "@/lib/copy";
import { occAgentEnabled } from "@/lib/features";

const NAV_ITEMS = [
  { to: "/", label: copy.nav.home, icon: Home, end: true },
  { to: "/reviews", label: copy.nav.review, icon: ListChecks, end: false },
  { to: "/workflows", label: copy.nav.workflows, icon: GitBranch, end: false },
  { to: "/rules", label: copy.nav.rules, icon: BookOpen, end: false },
  { to: "/history", label: copy.nav.history, icon: Clock, end: false },
  // A governed capability of its own, not a step of the delivery workflow.
  { to: "/onboarding", label: copy.nav.onboarding, icon: Building2, end: false },
  // The concentration-test review surface: client covenants become governed tests.
  { to: "/concentration", label: copy.nav.concentration, icon: Gauge, end: false },
];

/** The OCC Agent tab. Sits alongside the live tabs, behind its feature flag. */
const AGENT_ITEM = {
  to: "/agent",
  label: copy.nav.agent,
  icon: FlaskConical,
  end: false,
};

/** Administrator-only entry. Hidden for ordinary operators — the backend still
 *  refuses the route and every request behind it. */
const ADMIN_ITEM = {
  to: "/admin/config",
  label: copy.nav.admin,
  icon: SlidersHorizontal,
  end: false,
};

function NavItems({ compact }: { compact?: boolean }) {
  const { isAdmin } = useSession();
  const items = [
    ...NAV_ITEMS,
    ...(occAgentEnabled() ? [AGENT_ITEM] : []),
    ...(isAdmin ? [ADMIN_ITEM] : []),
  ];
  return (
    <>
      {items.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          className={({ isActive }) =>
            clsx(
              "flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-colors",
              compact && "shrink-0",
              isActive
                ? "bg-stone-900 text-white"
                : "text-stone-600 hover:bg-stone-100 hover:text-stone-900",
            )
          }
        >
          <Icon className="h-4 w-4" aria-hidden />
          <span>{label}</span>
        </NavLink>
      ))}
    </>
  );
}

function StartButton({ className }: { className?: string }) {
  return (
    <Link
      to="/new"
      className={clsx(
        "flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700",
        className,
      )}
    >
      <Plus className="h-4 w-4" aria-hidden />
      {copy.nav.startNew}
    </Link>
  );
}

/** App shell: left sidebar on desktop, top nav on small screens. */
export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-full flex-col md:flex-row">
      {/* Sidebar (desktop) */}
      <aside className="hidden w-60 shrink-0 flex-col border-r border-stone-200 bg-white px-4 py-6 md:flex">
        <Link to="/" className="mb-8 px-3 text-lg font-semibold tracking-tight text-stone-900">
          {copy.appName}
        </Link>
        <StartButton className="mb-6" />
        <nav className="flex flex-col gap-1">
          <NavItems />
        </nav>
      </aside>

      {/* Top bar (small screens) */}
      <header className="flex flex-col gap-3 border-b border-stone-200 bg-white px-4 py-3 md:hidden">
        <div className="flex items-center justify-between">
          <Link to="/" className="text-lg font-semibold tracking-tight text-stone-900">
            {copy.appName}
          </Link>
          <StartButton />
        </div>
        <nav className="flex gap-1 overflow-x-auto pb-1">
          <NavItems compact />
        </nav>
      </header>

      <main className="min-w-0 flex-1">{children}</main>
    </div>
  );
}
