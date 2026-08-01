import { Component, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle } from "lucide-react";
import { copy } from "@/lib/copy";

/**
 * A route-level safety net.
 *
 * A rendering fault inside the wrapped subtree shows a card and keeps the rest
 * of the shell standing. Without one, React unmounts the entire application on
 * the first render error — which is exactly the blank page this exists to
 * prevent. Retry resets the boundary and re-renders the subtree; nothing about
 * the case itself is touched, because rendering never writes.
 */

interface Props {
  children: ReactNode;
  /** Where "back" leads; defaults to the OCC Agent case list. */
  backTo?: string;
  backLabel?: string;
}

interface State {
  error: Error | null;
}

export class RouteErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error): void {
    // Surfaced for support: the operator-facing card stays non-technical.
    console.error("route error boundary caught", error);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="mx-auto max-w-2xl px-6 py-12" role="alert">
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" aria-hidden />
            <div>
              <h2 className="text-base font-semibold text-rose-900">
                {copy.agent.errorTitle}
              </h2>
              <p className="mt-1 text-sm text-rose-800">{copy.agent.errorBody}</p>
              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => this.setState({ error: null })}
                  className="rounded-xl bg-rose-700 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-800"
                >
                  {copy.agent.errorRetry}
                </button>
                <Link
                  to={this.props.backTo ?? "/agent"}
                  onClick={() => this.setState({ error: null })}
                  className="rounded-xl border border-rose-300 bg-white px-4 py-2 text-sm font-medium text-rose-800"
                >
                  {this.props.backLabel ?? copy.agent.errorBack}
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}
