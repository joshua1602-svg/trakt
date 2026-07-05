#!/usr/bin/env python3
"""Build a compact UK ITL3 SVG-path atlas from an ITL3/NUTS3 GeoJSON.

Reads a LOCAL GeoJSON (or a URL), projects lon/lat to an SVG viewBox
(equirectangular with a cos(lat) aspect correction), simplifies each ring with
Douglas-Peucker, and emits { viewBox, areas: { ITL3_CODE: {name, d} } }.

Purpose: turn the ONS full-resolution ITL3 Boundaries GeoJSON (~100MB) into a
tiny (~100-300KB) atlas suitable for a web choropleth. Pure stdlib — no installs.

Usage:
    python3 build_itl3_svg.py INPUT.geojson [OUTPUT.json]
    # INPUT may be a local path or an http(s) URL.

Handles the ONS field names across vintages: ITL325CD/NM (Jan 2025),
ITL321CD/NM (Jan 2021), or NUTS312CD/NM (2012, re-prefixed UK->TL).
"""
import json
import math
import sys
import urllib.request

# First CLI arg is the INPUT geojson (local path or URL); second is the OUTPUT.
INPUT = sys.argv[1] if len(sys.argv) > 1 else \
    "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/eurostat/ew/nuts3.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "uk_itl3_paths.json"
TARGET_W = 800.0
DP_EPS = 1.2       # simplification tolerance in projected px

# Property keys to try, most-recent vintage first: (code_key, name_key).
_PROP_KEYS = (("ITL325CD", "ITL325NM"), ("ITL321CD", "ITL321NM"),
              ("NUTS312CD", "NUTS312NM"), ("nuts312cd", "nuts312nm"))


def _prop(props, which):
    for code_key, name_key in _PROP_KEYS:
        key = code_key if which == "code" else name_key
        if props.get(key):
            return props[key]
    return ""


def _load(src):
    if src.startswith("http://") or src.startswith("https://"):
        return json.loads(urllib.request.urlopen(src, timeout=120).read())
    with open(src) as fh:
        return json.load(fh)


def nuts_to_itl(code: str) -> str:
    """Normalise to an ITL (TL...) code. ITL is already TL; NUTS 2012 is UK...."""
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
    gj = _load(INPUT)
    feats = [f for f in gj["features"] if f.get("geometry")]  # skip null-geometry rows
    if not feats:
        raise SystemExit(
            "No features with geometry. This looks like a 'Names and Codes' file "
            "(geometry: null) — download the ITL3 'Boundaries' dataset instead.")

    # Pass 1: extract every ring in source coords, applying the Shetland inset so
    # the far-north islands don't stretch the whole map with an empty sea gap.
    # (record[code] = {"name", "rings": [[(x, y), ...], ...]})
    records = {}
    for f in feats:
        props = f["properties"]
        code = nuts_to_itl(_prop(props, "code"))
        rec = records.setdefault(code, {"name": _prop(props, "name") or code, "rings": []})
        for poly in _polys(f["geometry"]):
            for ring in poly:
                rec["rings"].append([(x, y) for x, y in ring])

    _inset_shetland(records)

    # Global bbox over the (possibly inset) coordinates.
    xs = [p[0] for r in records.values() for ring in r["rings"] for p in ring]
    ys = [p[1] for r in records.values() for ring in r["rings"] for p in ring]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    # Auto-detect the coordinate system. lon/lat are small (|value| < 400); a
    # projected grid such as British National Grid (EPSG:27700, metres) is large
    # — use it planar (no cos-lat correction; BNG is already metric/equal-aspect).
    geographic = abs(x1) <= 400 and abs(y1) <= 400
    kx = math.cos(math.radians((y0 + y1) / 2)) if geographic else 1.0
    span_x = (x1 - x0) * kx
    span_y = (y1 - y0)
    scale = TARGET_W / span_x
    W = TARGET_W
    H = round(span_y * scale, 1)

    def project(px, py):
        x = (px - x0) * kx * scale
        y = (y1 - py) * scale  # flip so north is up
        return round(x, 1), round(y, 1)

    areas = {}
    for code, rec in records.items():
        parts = []
        for ring in rec["rings"]:
            proj = dp_simplify([project(x, y) for x, y in ring], DP_EPS)
            if len(proj) < 3:
                continue
            parts.append("M" + " ".join(f"{x},{y}" for x, y in proj) + "Z")
        if parts:
            areas[code] = {"name": rec["name"], "d": "".join(parts)}

    out = {"viewBox": f"0 0 {W:g} {H:g}", "areas": areas,
           "note": f"UK ITL3 boundaries, simplified from {INPUT.split('/')[-1]}. "
                   "Codes are ITL (TL...); source: ONS Open Geography Portal (OGL)."}
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    import os
    print(f"areas={len(areas)} viewBox='{out['viewBox']}' bytes={os.path.getsize(OUT):,}")


# Shetland inset (British National Grid, metres). Rings whose southernmost point
# is above the threshold are Shetland; translate them south/east into the empty
# North Sea beside the mainland so the map isn't stretched by the ~200km gap.
_SHETLAND_MIN_NORTHING = 1_090_000
_SHETLAND_DX = 140_000    # east, into open sea
_SHETLAND_DY = -330_000   # south, closing the gap


def _inset_shetland(records):
    """In-place: translate far-north Shetland rings into an inset position. No-op
    for lon/lat data (degrees) or when nothing qualifies."""
    pts = [p for r in records.values() for ring in r["rings"] for p in ring]
    if not pts or max(abs(p[1]) for p in pts) <= 800:  # lon/lat -> not BNG metres
        return
    moved = 0
    for rec in records.values():
        for i, ring in enumerate(rec["rings"]):
            if min(p[1] for p in ring) > _SHETLAND_MIN_NORTHING:
                rec["rings"][i] = [(x + _SHETLAND_DX, y + _SHETLAND_DY) for x, y in ring]
                moved += 1
    if moved:
        print(f"inset: moved {moved} Shetland ring(s)")


def _polys(geom):
    """Normalise Polygon/MultiPolygon to a list of polygons (each a list of rings)."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


if __name__ == "__main__":
    main()
