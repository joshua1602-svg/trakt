import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { TimingDisclosure } from "@/domain";
import { TimingDisclosureBanner } from "./TimingDisclosureBanner";

const info: TimingDisclosure = {
  fundedActualsAsOf: "2025-11-30",
  pipelineExtractAsOf: "2026-01-12",
  lagDays: 43,
  level: "info",
  message:
    "Funded actuals are as of 2025-11-30. Pipeline is shown using the latest weekly extract dated 2026-01-12.",
  warnThresholdDays: 45,
};

const warning: TimingDisclosure = {
  fundedActualsAsOf: "2025-11-30",
  pipelineExtractAsOf: "2026-01-20",
  lagDays: 51,
  level: "warning",
  message:
    "Pipeline extract is 51 days after the selected funded reporting date. Confirm funded actuals are pending before relying on forecast bridge metrics.",
  warnThresholdDays: 45,
};

const none: TimingDisclosure = {
  fundedActualsAsOf: "2025-11-30",
  pipelineExtractAsOf: "2025-11-10",
  lagDays: -20,
  level: "none",
  message: null,
  warnThresholdDays: 45,
};

describe("TimingDisclosureBanner", () => {
  it("renders the info disclosure with both anchors", () => {
    render(<TimingDisclosureBanner timing={info} />);
    const banner = screen.getByTestId("pipeline-timing-disclosure");
    expect(banner).toHaveAttribute("data-level", "info");
    expect(screen.getByText(/latest weekly extract dated 2026-01-12/)).toBeInTheDocument();
    expect(screen.getByText(/Funded actuals as of/)).toBeInTheDocument();
    expect(screen.getByText("2026-01-12")).toBeInTheDocument();
    expect(screen.getByText(/43-day lag/)).toBeInTheDocument();
  });

  it("renders the stronger warning above the threshold", () => {
    render(<TimingDisclosureBanner timing={warning} />);
    const banner = screen.getByTestId("pipeline-timing-disclosure");
    expect(banner).toHaveAttribute("data-level", "warning");
    expect(screen.getByText(/Confirm funded actuals are pending/)).toBeInTheDocument();
  });

  it("renders nothing when there is no lag to disclose", () => {
    const { container } = render(<TimingDisclosureBanner timing={none} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when timing is absent", () => {
    const { container } = render(<TimingDisclosureBanner timing={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });
});
