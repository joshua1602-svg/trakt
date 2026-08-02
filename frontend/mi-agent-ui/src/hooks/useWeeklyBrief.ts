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

export function useWeeklyBrief({ client, portfolioId, portfolioContext, asOf, enabled }: {
  client: AgentClient;
  portfolioId: string;
  portfolioContext?: string;
  asOf?: string;
  enabled: boolean;
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
    if (!enabled || !portfolioId) {
      setState(IDLE);
      return;
    }
    const token = ++latest.current;
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
  }, [portfolioId, portfolioContext, asOf, enabled]);

  return state;
}
