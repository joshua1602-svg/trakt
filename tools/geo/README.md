# UK ITL3 choropleth geometry

`build_itl3_svg.py` fetches UK ITL3 boundary polygons, projects them to an SVG
viewBox (equirectangular + cos(lat) aspect), simplifies each ring with
Douglas-Peucker, and emits a compact `{ viewBox, areas: { ITL3_CODE: {name, d} } }`
atlas for the frontend choropleth.

## IMPORTANT — boundary vintage
The MI tape uses **ITL3 January 2021** codes (`geographic_region_collateral_itl3`,
e.g. `TLK11` = Bristol). The pipeline MUST be built against **2021 ITL3**
boundaries so the codes align with the tape.

The authoritative source is the ONS Open Geography Portal
(International Territorial Level 3, January 2021, UK BUC/BGC). Point `URL` at that
GeoJSON (ArcGIS FeatureServer `f=geojson`, fields `ITL321CD`/`ITL321NM`) and set
`nuts_to_itl` to a pass-through.

The default `URL` here is a GitHub mirror of **NUTS3 2012** boundaries, used only
to validate the projection/simplification pipeline end-to-end. Its codes DO NOT
fully match the 2021 tape codes — do not ship its output as production geometry.
