#!/usr/bin/env python3
"""Build a compact UK ITL3 SVG-path atlas from NUTS3 GeoJSON.

Fetches England+Wales NUTS3 boundaries, projects lon/lat to an SVG viewBox
(equirectangular with a cos(lat) aspect correction), simplifies each ring with
Douglas-Peucker, and emits { viewBox, areas: { ITL3_CODE: {name, d} } }.

NUTS3 == ITL3 (same boundaries); codes only re-prefix UK... -> TL...
"""
import json
import math
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/eurostat/ew/nuts3.json"
TARGET_W = 800.0
DP_EPS = 1.2       # simplification tolerance in projected px
OUT = sys.argv[1] if len(sys.argv) > 1 else "uk_itl3_paths.json"


def nuts_to_itl(code: str) -> str:
    c = (code or "").strip().upper()
    return ("TL" + c[2:]) if c.startswith("UK") else c


def dp_simplify(pts, eps):
    """Iterative Douglas-Peucker (avoids recursion limits on big rings)."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    last = len(pts) - 1
    # A closed ring (first == last) has a zero-length baseline, so every point
    # measures distance 0 and DP would collapse it. Seed a second anchor: the
    # point farthest from the start, and simplify the two halves.
    ax, ay = pts[0]
    if math.isclose(ax, pts[last][0]) and math.isclose(ay, pts[last][1]):
        far, fd = 0, -1.0
        for i in range(1, last):
            d = math.hypot(pts[i][0] - ax, pts[i][1] - ay)
            if d > fd:
                fd, far = d, i
        if far:
            keep[far] = True
            stack = [(0, far), (far, last)]
        else:
            return [pts[0], pts[last]]
    else:
        stack = [(0, last)]
    while stack:
        a, b = stack.pop()
        ax, ay = pts[a]
        bx, by = pts[b]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy) or 1e-9
        dmax, idx = 0.0, -1
        for i in range(a + 1, b):
            px, py = pts[i]
            # perpendicular distance to segment a-b
            d = abs((px - ax) * dy - (py - ay) * dx) / norm
            if d > dmax:
                dmax, idx = d, i
        if dmax > eps and idx != -1:
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return [p for p, k in zip(pts, keep) if k]


def main():
    raw = urllib.request.urlopen(URL, timeout=60).read()
    gj = json.loads(raw)
    feats = gj["features"]

    # Global bbox for the projection.
    lons, lats = [], []
    for f in feats:
        for poly in _polys(f["geometry"]):
            for ring in poly:
                for lon, lat in ring:
                    lons.append(lon); lats.append(lat)
    lon0, lon1 = min(lons), max(lons)
    lat0, lat1 = min(lats), max(lats)
    mid_lat = math.radians((lat0 + lat1) / 2)
    kx = math.cos(mid_lat)
    span_x = (lon1 - lon0) * kx
    span_y = (lat1 - lat0)
    scale = TARGET_W / span_x
    W = TARGET_W
    H = round(span_y * scale, 1)

    def project(lon, lat):
        x = (lon - lon0) * kx * scale
        y = (lat1 - lat) * scale  # flip so north is up
        return round(x, 1), round(y, 1)

    areas = {}
    for f in feats:
        props = f["properties"]
        code = nuts_to_itl(props.get("NUTS312CD") or props.get("nuts312cd") or "")
        name = props.get("NUTS312NM") or props.get("nuts312nm") or code
        parts = []
        for poly in _polys(f["geometry"]):
            for ring in poly:
                proj = [project(lon, lat) for lon, lat in ring]
                proj = dp_simplify(proj, DP_EPS)
                if len(proj) < 3:
                    continue
                d = "M" + " ".join(f"{x},{y}" for x, y in proj) + "Z"
                parts.append(d)
        if parts:
            areas[code] = {"name": name, "d": "".join(parts)}

    out = {"viewBox": f"0 0 {W:g} {H:g}", "areas": areas,
           "note": "England & Wales ITL3 (NUTS3) boundaries, simplified. "
                   "Codes re-prefixed UK->TL. Source: martinjc/UK-GeoJSON (Eurostat)."}
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    import os
    print(f"areas={len(areas)} viewBox='{out['viewBox']}' bytes={os.path.getsize(OUT):,}")


def _polys(geom):
    """Normalise Polygon/MultiPolygon to a list of polygons (each a list of rings)."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


if __name__ == "__main__":
    main()
