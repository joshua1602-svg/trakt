/**
 * The client control shows the CLIENT'S NAME, and shows it as approved.
 *
 * Reported defect: this control read "Platform" — an internal placeholder — and
 * before that "Client 001", an identifier. The API now sends the governed name
 * alongside the label (see mi_agent_api/client_identity.py).
 *
 * The browser's job here is only to tell a name from an identifier and render
 * each correctly. It never derives a name, and it never re-cases one: a client
 * approved "ERE Funding - Equity Release Mortgages", not "Ere Funding Equity
 * Release Mortgages", which is what title-casing a lowercased copy produces.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { PortfolioSelector } from "./PortfolioSelector";
import type { SnapshotPortfolio } from "@/domain";

const GOVERNED = "ERE Funding - Equity Release Mortgages";

function portfolio(over: Partial<SnapshotPortfolio> = {}): SnapshotPortfolio {
  return { client_id: "client_001", label: "CLIENT_001", runs: [], ...over };
}

describe("PortfolioSelector", () => {
  it("renders the governed name exactly as it was approved", () => {
    render(<PortfolioSelector portfolios={[portfolio({ client_name: GOVERNED })]}
      value="client_001" onChange={() => {}} />);
    expect(screen.getByText(GOVERNED)).toBeInTheDocument();
    // Not the title-cased mangling, and not the identifier.
    expect(screen.queryByText("Ere Funding Equity Release Mortgages")).toBeNull();
    expect(screen.queryByText("Client 001")).toBeNull();
  });

  it("still prettifies an identifier when no name is governed", () => {
    render(<PortfolioSelector portfolios={[portfolio()]} value="client_001"
      onChange={() => {}} />);
    expect(screen.getByText("Client 001")).toBeInTheDocument();
  });

  it("never shows the platform placeholder as a client", () => {
    // What the header actually read on a blob-triggered run.
    render(<PortfolioSelector portfolios={[portfolio({ client_name: GOVERNED, label: "PLATFORM" })]}
      value="client_001" onChange={() => {}} />);
    expect(screen.queryByText(/^platform$/i)).toBeNull();
    expect(screen.getByText(GOVERNED)).toBeInTheDocument();
  });

  it("treats a blank governed name as no name at all", () => {
    render(<PortfolioSelector portfolios={[portfolio({ client_name: "   " })]}
      value="client_001" onChange={() => {}} />);
    expect(screen.getByText("Client 001")).toBeInTheDocument();
  });

  it("carries the full name as a tooltip, since a long name truncates", () => {
    render(<PortfolioSelector portfolios={[portfolio({ client_name: GOVERNED })]}
      value="client_001" onChange={() => {}} />);
    expect(screen.getByTitle(GOVERNED)).toBeInTheDocument();
  });

  it("names each client in the picker, and switches on the client id", () => {
    const onChange = vi.fn();
    render(<PortfolioSelector
      portfolios={[
        portfolio({ client_name: GOVERNED }),
        portfolio({ client_id: "beta_bank", label: "BETA_BANK", client_name: "Beta Bank NV" }),
      ]}
      value="client_001" onChange={onChange} />);
    fireEvent.click(screen.getByText(GOVERNED));
    fireEvent.click(screen.getByText("Beta Bank NV"));
    expect(onChange).toHaveBeenCalledWith("beta_bank");
  });
});
