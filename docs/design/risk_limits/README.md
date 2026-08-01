# Risk Limits — three-state workspace wireframes

Design artefacts for the Funded → Risk Limits iteration that adds the
three-state concentration evaluation:

| State | Meaning | Never called |
|---|---|---|
| **Funded** | The contractual compliance position — governed funded portfolio only, as of the reporting date. | "current estimate" |
| **Expected Forecast** | Funded + the statistically expected contribution of the current pipeline, from the existing completion-trend model (per-stage empirical rates over the weekly-snapshot observation window, 12-observation sufficiency floor, config fallback). | just "forecast" without the methodology link |
| **Full Pipeline** | Funded + 100% of all active in-scope pipeline. A deliberately unrealistic **maximum-exposure stress**, non-probability-weighted. | "forecast", "expected", "projection" |

Files:

* `01_desktop_summary.md` — header, summary tiles, emerging risks.
* `02_desktop_table.md` — the three-state comparison table + filters.
* `03_detail_panel.md` — per-test detail: values, semantics, drivers, provenance.
* `04_pipeline_drivers.md` — driver drill-through.
* `05_methodology_panel.md` — forecast methodology disclosure.
* `06_mobile_and_states.md` — narrow-width layout; loading/empty/unavailable/error;
  the three breach archetypes (funded breach, expected breach,
  full-pipeline-only breach).
* `07_component_hierarchy.md` — annotated component tree, interaction notes,
  status hierarchy, accessibility and responsive behaviour.

Conventions used in the wireframes: `[Badge:tone]` refers to the existing
`Badge` primitive tones (mint/amber/rose/navy/neutral); glyphs `✓ ! ✕ – …`
are the existing non-colour status cues; boxes are drawn to structure, not to
scale. All numbers shown are the worked South West example from
`docs/concentration_tests.md` phase-2 sections.

## Recommended layout (summary)

One page, four bands: (A) header with configuration + methodology provenance,
(B) six summary tiles split *by state* (funded / expected / stress), (C) an
**Emerging risks** band of ranked, deterministic statements, (D) the
three-state table with a single risk classification column; selecting a row
opens the existing detail panel extended with a state comparison strip,
drivers table and methodology accordion. Funded remains the anchor state:
it is always the first value column and the only one used for the historical
trend. Expected Forecast carries the movement arrow from Funded; Full
Pipeline is visually de-emphasised (muted column header, "stress" sublabel)
so a stress number is never mistaken for a prediction.
