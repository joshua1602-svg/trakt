import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { WatchlistItem } from "@/domain";
import { PipelineWatchlist } from "./PipelineWatchlist";

const ITEMS: WatchlistItem[] = [
  {
    category: "broker_channel_concentration",
    severity: "warning",
    title: "Broker concentration: Broker Alpha is 45% of pipeline",
    detail: "Top broker 'Broker Alpha' accounts for 45.0% (£560,000) of £1,250,000 pipeline amount.",
  },
  {
    category: "missing_completion_probability",
    severity: "info",
    title: "1 case without a completion probability",
    detail: "1/10 rows have no row or config stage probability; excluded from weighted forecast.",
  },
];

describe("PipelineWatchlist", () => {
  it("renders concise business-facing warning titles", () => {
    render(<PipelineWatchlist items={ITEMS} />);
    expect(screen.getByText("Pipeline Watchlist")).toBeInTheDocument();
    expect(screen.getByText(/Broker concentration: Broker Alpha is 45%/)).toBeInTheDocument();
    expect(screen.getByText(/1 case without a completion probability/)).toBeInTheDocument();
  });

  it("keeps the technical detail hidden until expanded", () => {
    render(<PipelineWatchlist items={ITEMS} />);
    expect(screen.queryByText(/accounts for 45.0%/)).not.toBeInTheDocument();
    fireEvent.click(screen.getAllByLabelText("Toggle technical detail")[0]);
    expect(screen.getByText(/accounts for 45.0%/)).toBeInTheDocument();
  });

  it("renders an empty state with no warnings", () => {
    render(<PipelineWatchlist items={[]} />);
    expect(screen.getByText("No early warnings for this run.")).toBeInTheDocument();
  });

  it("shows withdrawn exclusions as INFO with a clear business message", () => {
    const items: WatchlistItem[] = [
      {
        category: "withdrawn_excluded_from_weighting",
        severity: "info",
        title: "106 withdrawn/inactive case(s) excluded from forecast probability weighting",
        detail: "By stage [WITHDRAWN:106]. Intentionally excluded from weighted forecast.",
        excluded: true,
      },
    ];
    render(<PipelineWatchlist items={items} />);
    // Business message is clear (not the old ambiguous "without a completion probability").
    expect(screen.getByText(/withdrawn\/inactive case\(s\) excluded from forecast/i)).toBeInTheDocument();
    expect(screen.queryByText(/without a completion probability/i)).not.toBeInTheDocument();
    // Severity badge reads "info", not "warning".
    expect(screen.getByText("info")).toBeInTheDocument();
    expect(screen.queryByText("warning")).not.toBeInTheDocument();
  });
});
