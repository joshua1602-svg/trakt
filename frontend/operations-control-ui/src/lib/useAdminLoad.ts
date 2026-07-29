import { useCallback, useEffect, useRef, useState } from "react";
import { OpsError } from "@/api/OpsClient";
import { errorMessage } from "./useLoad";

interface AdminLoadState<T> {
  data?: T;
  error?: string;
  /** The backend refused this administrator area for this user. */
  forbidden: boolean;
  loading: boolean;
}

/**
 * Load-on-mount for administrator data, separating "refused" from "failed".
 *
 * A 403 from the backend is the real access decision — the client-side role
 * check only decides what to offer, so every administrator screen still has to
 * handle being refused here.
 */
export function useAdminLoad<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<AdminLoadState<T>>({ loading: true, forbidden: false });
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const generation = useRef(0);

  const reload = useCallback(async (options?: { quiet?: boolean }) => {
    const gen = ++generation.current;
    if (!options?.quiet) {
      setState((prev) => ({ ...prev, loading: true, error: undefined, forbidden: false }));
    }
    try {
      const data = await fnRef.current();
      if (gen === generation.current) {
        setState({ data, loading: false, forbidden: false });
      }
    } catch (err) {
      if (gen === generation.current) {
        setState((prev) => ({
          ...prev,
          loading: false,
          forbidden: err instanceof OpsError && err.isForbidden,
          error: errorMessage(err),
        }));
      }
    }
  }, []);

  useEffect(() => {
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { ...state, reload };
}
