# UK ITL3 choropleth geometry

`build_itl3_svg.py` turns an ONS ITL3 **Boundaries** GeoJSON into a compact
`{ viewBox, areas: { ITL3_CODE: {name, d} } }` SVG-path atlas for the frontend
choropleth. It projects lon/lat to an SVG viewBox (equirectangular + cos(lat)
aspect), simplifies each ring with Douglas-Peucker, and rounds coordinates —
taking the ONS full-resolution file (~100MB) down to ~100-300KB.

**Pure Python standard library — no pip installs.**

## The problem it solves
The ONS full-resolution ITL3 Boundaries GeoJSON is ~100MB — too big to upload or
commit. A web choropleth needs almost none of that detail. Run this **locally**
on the big file; commit/upload only the tiny atlas it produces.

## Usage
```bash
python3 build_itl3_svg.py INPUT.geojson OUTPUT.json
# INPUT  = the ONS ITL3 Boundaries GeoJSON (local path or URL)
# OUTPUT = the compact atlas to commit (e.g. uk_itl3_paths.json, ~100-300KB)
```

## Sourcing the input (matches the tape)
The tape uses **ITL3 January 2025** codes (confirmed: 182/182 match
`uk_itl_master_lookup_v2.csv`). From the ONS Open Geography Portal
(geoportal.statistics.gov.uk), download the dataset titled
**"International Territorial Level 3 (January 2025) Boundaries UK BUC"** as
GeoJSON — the **Boundaries** product (features have real `geometry`), NOT the
**Names and Codes** product (every `geometry` is `null`). BUC = Ultra Generalised
(smallest); BGC also fine. Field names `ITL325CD` / `ITL325NM` are handled
automatically (as are the 2021 `ITL321*` and 2012 `NUTS312*` variants).

Open Government Licence — free to bundle with attribution.
