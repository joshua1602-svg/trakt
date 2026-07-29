import { createContext, useContext, type ReactNode } from "react";
import type { Principal } from "./adminTypes";
import { useOpsClient } from "./context";
import { useLoad } from "@/lib/useLoad";

interface Session {
  principal?: Principal;
  loading: boolean;
  /** True only when the backend has confirmed the administrator role. */
  isAdmin: boolean;
}

const SessionContext = createContext<Session | null>(null);

/**
 * Loads the signed-in principal once, so the shell knows what to OFFER.
 *
 * This is a convenience, never a control: every administrator route and every
 * administrator request is authorised again by the backend, and a 403 is
 * handled wherever it lands.
 */
export function SessionProvider({ children }: { children: ReactNode }) {
  const client = useOpsClient();
  const { data, loading } = useLoad(() => client.getPrincipal(), []);
  return (
    <SessionContext.Provider
      value={{ principal: data, loading, isAdmin: Boolean(data?.is_admin) }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession(): Session {
  const session = useContext(SessionContext);
  if (!session) {
    throw new Error("SessionProvider is missing above this component");
  }
  return session;
}
