/**
 * Phase 2A — fetch the optional movement detail for ONE weekly point.
 *
 * The discipline this hook enforces matters more than what it returns:
 *
 *  * nothing is requested unless the feature is enabled AND a point is actually
 *    selected — a chart that is merely rendered issues no request;
 *  * the selected point is DEBOUNCED, so a pointer sweeping across a 26-week
 *    series triggers one request for where it settles, not one per week;
 *  * the client cache is keyed by week, so re-hovering a point never refetches
 *    and in-flight requests for the same point are shared;
 *  * a rejection is "no detail", never an error the user has to see. The chart's
 *    existing tooltip is always the fallback.
 */

import { useEffect, useRef, useState } from "react";
import type { AgentClient } from "@/api";
import type { MovementDetail, MovementDetailType } from "@/domain";

/** How long the pointer must settle on a point before anything is requested. */
export const HOVER_DEBOUNCE_MS = 220;

export interface MovementDetailState {
  detail: MovementDetail | null;
  loading: boolean;
  /** True when the request completed but there is nothing to show. */
  unavailable: boolean;
}

const IDLE: MovementDetailState = { detail: null, loading: false, unavailable: false };

export interface UseMovementDetailArgs {
  client: AgentClient;
  portfolioId: string;
  detailType: MovementDetailType;
  /** The hovered / focused week, or null when nothing is selected. */
  asOf: string | null;
  portfolioContext?: string;
  enabled: boolean;
  debounceMs?: number;
}

export function useMovementDetail(
  { client, portfolioId, detailType, asOf, portfolioContext, enabled,
    debounceMs = HOVER_DEBOUNCE_MS }: UseMovementDetailArgs,
): MovementDetailState {
  const [state, setState] = useState<MovementDetailState>(IDLE);
  // Identifies the request the effect is allowed to publish. A slower earlier
  // request resolving after a newer one must not overwrite it.
  const latest = useRef(0);

  useEffect(() => {
    if (!enabled || !asOf || !portfolioId) {
      setState(IDLE);
      return;
    }
    const token = ++latest.current;
    const timer = setTimeout(() => {
      setState((prev) => ({ ...prev, loading: true }));
      Promise.resolve(
        client.getMovementDetail?.(portfolioId, detailType, asOf, portfolioContext),
      )
        .then((detail) => {
          if (token !== latest.current) return;
          setState({
            detail: detail ?? null,
            loading: false,
            unavailable: !detail || detail.available === false,
          });
        })
        .catch(() => {
          // A 404 (feature off on the API), a network blip or an abort all mean
          // the same thing to the user: no detail. Never a visible error.
          if (token !== latest.current) return;
          setState({ detail: null, loading: false, unavailable: true });
        });
    }, debounceMs);

    return () => {
      clearTimeout(timer);
    };
  }, [client, portfolioId, detailType, asOf, portfolioContext, enabled, debounceMs]);

  return state;
}
