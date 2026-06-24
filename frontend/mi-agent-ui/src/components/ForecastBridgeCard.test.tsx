import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ForecastBridgeCard } from "./ForecastBridgeCard";
import { mockForecastSnapshot } from "@/data/mockForecast";

describe("ForecastBridgeCard", () => {
  const bridge = mockForecastSnapshot("client_001/mi_2025_11").forecastBridge!;

  it("renders funded + weighted pipeline = forecast (deterministic, backend-derived)", () => {
    render(<ForecastBridgeCard bridge={bridge} />);
    expect(screen.getByText("Funded + Pipeline Forecast")).toBeInTheDocument();
    expect(screen.getByText("Current funded balance")).toBeInTheDocument();
    expect(screen.getByText("Weighted expected pipeline")).toBeInTheDocument();
    expect(screen.getByText("Forecast funded balance")).toBeInTheDocument();
    // £8.9MM funded + £1.1MM weighted = £10.0MM forecast (compact GBP).
    expect(screen.getByText("£8.9MM")).toBeInTheDocument();
    expect(screen.getByText("£1.1MM")).toBeInTheDocument();
    expect(screen.getByText("£10.0MM")).toBeInTheDocument();
  });

  it("shows the completion probability basis and readiness status", () => {
    render(<ForecastBridgeCard bridge={bridge} />);
    expect(screen.getByText(/mixed_historical_and_config/)).toBeInTheDocument();
    expect(screen.getByText(/Forecast ready/i)).toBeInTheDocument();
  });

  it("discloses blended weighted conversion and excluded amount", () => {
    render(<ForecastBridgeCard bridge={bridge} />);
    expect(screen.getByText(/Blended weighted conversion/i)).toBeInTheDocument();
    expect(screen.getByText(/64\.1%/)).toBeInTheDocument();
    expect(screen.getByText(/Excluded from weighting/i)).toBeInTheDocument();
  });

  it("surfaces readiness warnings for a partial forecast", () => {
    const partial = mockForecastSnapshot("client_001/mi_2025_10").forecastBridge!;
    render(<ForecastBridgeCard bridge={partial} />);
    expect(screen.getByText(/Forecast partial/i)).toBeInTheDocument();
    expect(screen.getByText(/expected completion date present but empty/i)).toBeInTheDocument();
  });

  it("renders nothing when no bridge is available", () => {
    const { container } = render(<ForecastBridgeCard bridge={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
