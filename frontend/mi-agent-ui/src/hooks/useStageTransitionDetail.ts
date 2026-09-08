/**
 * Sprint 2 — fetch the governed GROSS stage-transition detail for one window.
 *
 * The same request discipline as `useMovementDetail`, and for the same reasons:
 *
 *  * nothing is requested unless the feature is enabled and a portfolio is
 *    selected — a panel that is merely mounted behind a closed tab issues no
 *    request;
 *  * a slower earlier response can never overwrite a newer one;
 *  * a rejection is "no detail", never an error the user has to see.
 *
 * It differs from `useMovementDetail` in exactly one respect: this is a PANEL,
 * not a hover, so there is no pointer to settle and no debounce. Reusing the
 * hover hook would have meant either a spurious delay before the panel filled
 * in, or a debounce parameter that every caller had to remember to disable.
 *
 * It computes nothing. Every count, amount, outcome and residual on screen is
 * the engine's, fetched through the SAME `/mi/insight/movement-detail` route
 * the movement hover already uses.
 */

import { useEffect, useState } from "react";
import type { AgentClient } from "@/api";
import type { StageTransitionDetail } from "@/domain";

export interface StageTransitionState {
  detail: StageTransitionDetail | null;
  loading: boolean;
  /** True when the request completed but there is nothing to show. */
  unavailable: boolean;
}

const IDLE: StageTransitionState = {
  detail: null, loading: false, unavailable: false,
};

export interface UseStageTransitionDetailArgs {
  client: AgentClient;
  portfolioId: string;
  /** The week to explain, or null / undefined for the latest governed pair. */
  asOf?: string | null;
  portfolioContext?: string;
  enabled: boolean;
}

export function useStageTransitionDetail(
  { client, portfolioId, asOf, portfolioContext, enabled }:
    UseStageTransitionDetailArgs,
): StageTransitionState {
  const [state, setState] = useState<StageTransitionState>(IDLE);

  useEffect(() => {
    if (!enabled || !portfolioId) {
      setState(IDLE);
      return;
    }
    let live = true;
    setState((prev) => ({ ...prev, loading: true }));
    Promise.resolve(
      client.getStageTransitionDetail?.(portfolioId, asOf ?? undefined,
                                        portfolioContext),
    )
      .then((detail) => {
        if (!live) return;
        setState({
          detail: detail ?? null,
          loading: false,
          // The ENGINE decides availability. A duplicate identifier is a
          // governed refusal, and treating it as an empty result here would
          // render "nothing moved" over a window the engine declined to answer.
          unavailable: !detail || detail.available === false,
        });
      })
      .catch(() => {
        // A 404 (layer off on the API), a network blip or an abort all mean the
        // same thing to the reader: no detail. Never a visible error.
        if (!live) return;
        setState({ detail: null, loading: false, unavailable: true });
      });

    return () => { live = false; };
  }, [client, portfolioId, asOf, portfolioContext, enabled]);

  return state;
}
