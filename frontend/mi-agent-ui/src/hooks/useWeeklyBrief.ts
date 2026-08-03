/**
 * Phase 3A — fetch the Weekly Portfolio Brief.
 *
 * One request per portfolio + scope + week. The client cache keys on exactly
 * that, so a re-render never refetches and two components asking at once share
 * one in-flight request.
 *
 * A rejection is "no brief", never an error the user has to see: the API flag
 * may be off while the UI flag is on, and that must degrade to the current
 * dashboard rather than to a red panel.
 *
 * The brief NEVER competes with the dashboard's own first load. It summarises
 * data the dashboard is already fetching, so firing it in parallel means paying
 * for the same cold work twice, at once, on a worker that is simultaneously
 * trying to serve the primary requests — which is exactly what made the landing
 * page slow. It waits for `ready`, then for the main thread to go idle.
 */

import { useEffect, useRef, useState } from "react";
import type { AgentClient } from "@/api";
import type { WeeklyBrief } from "@/domain";

export interface WeeklyBriefState {
  brief: WeeklyBrief | null;
  loading: boolean;
  /** The request completed but there is nothing to show. */
  unavailable: boolean;
}

const IDLE: WeeklyBriefState = { brief: null, loading: false, unavailable: false };

/** Wait for the main thread to go quiet, so the fetch cannot land in the middle
 * of the chart render that follows the primary data. Falls back to a timeout
 * where requestIdleCallback is unavailable (Safari, jsdom). */
function whenIdle(fn: () => void, timeoutMs = 2000): () => void {
  const w = globalThis as unknown as {
    requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number;
    cancelIdleCallback?: (h: number) => void;
  };
  if (typeof w.requestIdleCallback === "function") {
    const handle = w.requestIdleCallback(fn, { timeout: timeoutMs });
    return () => w.cancelIdleCallback?.(handle);
  }
  const t = setTimeout(fn, 0);
  return () => clearTimeout(t);
}

export function useWeeklyBrief({ client, portfolioId, portfolioContext, asOf, enabled,
                                 ready = true }: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  asOf?: string;
  enabled: boolean;
  /**
   * Whether the dashboard's own primary data has resolved.
   *
   * Defaults to true so a caller that does not care — and every existing test —
   * behaves as before. `AppShell` passes the real signal, which is what keeps
   * the brief off the first-paint critical path.
   */
  ready?: boolean;
}): WeeklyBriefState {
  const [state, setState] = useState<WeeklyBriefState>(IDLE);
  // Guards against a slower earlier request resolving after a newer one and
  // overwriting the brief for the portfolio the user has since switched to.
  const latest = useRef(0);
  // The client is held in a ref and kept OUT of the effect's dependencies.
  // "One request per portfolio, scope and week" then holds however the caller
  // memoises: a parent that rebuilds its client wrapper on every render can no
  // longer turn a re-render into a refetch.
  const clientRef = useRef(client);
  clientRef.current = client;

  useEffect(() => {
    if (!enabled || !ready || !portfolioId) {
      // Not "loading": the panel renders nothing until there is something to
      // say, so a brief that has not started must not occupy the page either.
      setState(IDLE);
      return;
    }
    const token = ++latest.current;
    const cancel = whenIdle(() => {
      if (token !== latest.current) return;
      setState((prev) => ({ ...prev, loading: true }));
      Promise.resolve(
        clientRef.current.getWeeklyBrief?.(portfolioId, portfolioContext, asOf),
      )
        .then((brief) => {
          if (token !== latest.current) return;
          setState({
            brief: brief ?? null,
            loading: false,
            unavailable: !brief || brief.status === "unavailable",
          });
        })
        .catch(() => {
          // A 404 (feature off on the API), a network blip or an abort all mean
          // the same thing here: no brief. Never a visible error.
          if (token !== latest.current) return;
          setState({ brief: null, loading: false, unavailable: true });
        });
    });
    return cancel;
  }, [portfolioId, portfolioContext, asOf, enabled, ready]);

  return state;
}
