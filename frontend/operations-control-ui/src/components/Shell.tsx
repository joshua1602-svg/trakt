import type { ReactNode } from "react";
import { NavLink, Link } from "react-router-dom";
import {
  Home,
  ListChecks,
  GitBranch,
  BookOpen,
  Clock,
  Building2,
  Bot,
  Plus,
  Send,
  SlidersHorizontal,
  Wrench,
  Gauge,
} from "lucide-react";
import clsx from "clsx";
import { useSession } from "@/api/session";
import { copy } from "@/lib/copy";
import { occAgentEnabled } from "@/lib/features";

interface NavEntry {
  to: string;
  label: string;
  icon: typeof Home;
  end: boolean;
}

/** The manual-first navigation, unchanged for environments without the agent. */
const MANUAL_FIRST_NAV: NavEntry[] = [
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

/**
 * The agent-first hierarchy. The OCC Agent is the operating entry point;
 * manual delivery stays reachable below as the specialist route — for
 * exceptions, recovery, legacy cases and direct operator control.
 */
const AGENT_FIRST_NAV: NavEntry[] = [
  { to: "/", label: copy.nav.home, icon: Home, end: true },
  { to: "/agent", label: copy.nav.agent, icon: Bot, end: false },
  { to: "/onboarding", label: copy.nav.onboarding, icon: Building2, end: false },
  // Onboarding produces the proposals; this is where they are decided.
  { to: "/concentration", label: copy.nav.concentration, icon: Gauge, end: false },
  { to: "/reviews", label: copy.nav.review, icon: ListChecks, end: false },
  { to: "/workflows", label: copy.nav.workflows, icon: GitBranch, end: false },
  { to: "/history", label: copy.nav.history, icon: Clock, end: false },
];

const AGENT_FIRST_SPECIALIST: NavEntry[] = [
  { to: "/new", label: copy.nav.manual, icon: Wrench, end: false },
  { to: "/rules", label: copy.nav.rules, icon: BookOpen, end: false },
];

/** Administrator-only entry. Hidden for ordinary operators — the backend still
 *  refuses the route and every request behind it. */
const ADMIN_ITEM: NavEntry = {
  to: "/admin/config",
  label: copy.nav.admin,
  icon: SlidersHorizontal,
  end: false,
};

function NavItem({ to, label, icon: Icon, end, compact }: NavEntry & { compact?: boolean }) {
  return (
    <NavLink
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
  );
}

function NavItems({ compact }: { compact?: boolean }) {
  const { isAdmin } = useSession();
  if (!occAgentEnabled()) {
    const items = [...MANUAL_FIRST_NAV, ...(isAdmin ? [ADMIN_ITEM] : [])];
    return (
      <>
        {items.map((item) => (
          <NavItem key={item.to} {...item} compact={compact} />
        ))}
      </>
    );
  }
  const specialist = [...AGENT_FIRST_SPECIALIST, ...(isAdmin ? [ADMIN_ITEM] : [])];
  return (
    <>
      {AGENT_FIRST_NAV.map((item) => (
        <NavItem key={item.to} {...item} compact={compact} />
      ))}
      {!compact && (
        <p className="mt-4 px-3 text-xs font-semibold uppercase tracking-wide text-stone-400">
          {copy.nav.specialistHeading}
        </p>
      )}
      {specialist.map((item) => (
        <NavItem key={item.to} {...item} compact={compact} />
      ))}
    </>
  );
}

/** The primary start action: the OCC Agent when enabled, manual otherwise. */
function StartButton({ className }: { className?: string }) {
  const agentFirst = occAgentEnabled();
  return (
    <Link
      to={agentFirst ? "/agent" : "/new"}
      className={clsx(
        "flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700",
        className,
      )}
    >
      {agentFirst ? (
        <Send className="h-4 w-4" aria-hidden />
      ) : (
        <Plus className="h-4 w-4" aria-hidden />
      )}
      {agentFirst ? copy.nav.startAgent : copy.nav.startNew}
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
