import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import clsx from "clsx";
import { useSession } from "@/api/session";
import { AccessDenied } from "@/components/admin/AccessDenied";
import { Loading } from "@/components/ErrorNote";
import { copy } from "@/lib/copy";

const TABS = [
  { to: "/admin/config", label: copy.admin.tabs.overview, end: true },
  { to: "/admin/config/system", label: copy.admin.tabs.system, end: false },
  { to: "/admin/config/assets", label: copy.admin.tabs.assets, end: false },
  { to: "/admin/config/regimes", label: copy.admin.tabs.regimes, end: false },
  { to: "/admin/config/history", label: copy.admin.tabs.history, end: false },
];

export function AdminTabs() {
  return (
    <nav className="mb-8 flex gap-1 overflow-x-auto border-b border-stone-200 pb-px">
      {TABS.map((tab) => (
        <NavLink
          key={tab.to}
          to={tab.to}
          end={tab.end}
          className={({ isActive }) =>
            clsx(
              "shrink-0 border-b-2 px-4 py-2 text-sm font-medium transition-colors",
              isActive
                ? "border-stone-900 text-stone-900"
                : "border-transparent text-stone-500 hover:text-stone-800",
            )
          }
        >
          {tab.label}
        </NavLink>
      ))}
    </nav>
  );
}

/**
 * Route guard for the administrator area.
 *
 * The check here decides what to render; the backend decides what is allowed.
 * A non-administrator who types the route directly gets the plain-English
 * refusal, and any 403 the backend returns lands on the same screen.
 */
export function AdminOnly({ children }: { children: ReactNode }) {
  const { isAdmin, loading } = useSession();
  if (loading) return <Loading />;
  if (!isAdmin) return <AccessDenied />;
  return <>{children}</>;
}
