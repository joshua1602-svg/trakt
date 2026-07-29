import { useCallback, useEffect, useRef, useState } from "react";
import { copy } from "./copy";

interface LoadState<T> {
  data?: T;
  error?: string;
  loading: boolean;
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return copy.errors.generic;
}

/**
 * Small load-on-mount helper. `reload({ quiet: true })` refreshes without
 * flipping back into the loading state (used for polling).
 */
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<LoadState<T>>({ loading: true });
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const generation = useRef(0);

  const reload = useCallback(async (options?: { quiet?: boolean }) => {
    const gen = ++generation.current;
    if (!options?.quiet) {
      setState((prev) => ({ ...prev, loading: true, error: undefined }));
    }
    try {
      const data = await fnRef.current();
      if (gen === generation.current) {
        setState({ data, loading: false });
      }
    } catch (err) {
      if (gen === generation.current) {
        setState((prev) => ({ ...prev, loading: false, error: errorMessage(err) }));
      }
    }
  }, []);

  useEffect(() => {
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { ...state, reload };
}
