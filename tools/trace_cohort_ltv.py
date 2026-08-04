#!/usr/bin/env python3
"""Trace a cohort's WA LTV through every layer, to the first point it goes wrong.

Written for the reported case — direct_001 / vintage 2025-10 / +8 months showing
~0.4% — and usable against any approved snapshot without changing methodology:
it only READS.

    python tools/trace_cohort_ltv.py --root <output-root> --client <id> \
        --scope direct_001 --vintage 2025-10 --grain M

It prints, per layer, the value and the unit it is in:

  1. source canonical      the raw column, its storage scale and a sample
  2. prepared frame        after funded_prep (which normalises LTV to a fraction)
  3. cohort frame          after the static-pool membership filter
  4. shared service        the funded_cohort_progression payload
  5. React render          payload x100, exactly as pctFmt does
  6. PPTX render           payload through the deck's own _pct

A layer whose value is ~100x its predecessor's is the first point of failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LTV = "current_loan_to_value"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="onboarding output root")
    ap.add_argument("--client", required=True)
    ap.add_argument("--scope", default=None,
                    help="source_portfolio_id to narrow to (e.g. direct_001)")
    ap.add_argument("--vintage", default=None, help="e.g. 2025-10")
    ap.add_argument("--grain", default="M", choices=["Y", "Q", "M"])
    ap.add_argument("--period", default=None,
                    help="reporting date to focus on; default = every period")
    args = ap.parse_args()

    from mi_agent_api import evolution
    from mi_agent_api.evolution import (_origination_labels, _pct_fraction,
                                        _weighted_avg, percent_storage_scale)

    lens = {"source_portfolio_id": args.scope} if args.scope else None
    frames = evolution.funded_frames(args.root, args.client, None)
    if not frames:
        print("no funded reporting periods discovered under that root")
        return 2

    print(f"cohort: scope={args.scope or 'Total'} vintage={args.vintage} "
          f"grain={args.grain}\n")
    for fr in frames:
        date = fr.get("reporting_date")
        if args.period and date != args.period:
            continue
        df = fr["df"]
        # 1/2. the prepared frame IS what funded_frames returns; report the scale.
        scale = percent_storage_scale(df[LTV]) if LTV in df.columns else "absent"
        sample = df[LTV].dropna().head(3).tolist() if LTV in df.columns else []
        d = evolution._scope_frame_lens(df, lens)
        if args.vintage and d is not None:
            labels = _origination_labels(d, args.grain)
            d = d.iloc[0:0] if labels is None else d[labels.astype(str) == args.vintage]
        raw_wavg = _weighted_avg(d, LTV) if d is not None and len(d) else None
        frac = _pct_fraction(d, LTV) if d is not None and len(d) else None
        print(f"  {date}  rows={0 if d is None else len(d):>6}")
        print(f"      2. prepared    scale={scale}  sample={sample}")
        print(f"      3. cohort      weighted_avg={raw_wavg}")
        print(f"      4. service     wa_ltv={frac}   (contract unit: FRACTION)")
        if frac is not None:
            from mi_agent_pptx.cohorts import _pct
            print(f"      5. React       {frac * 100:.4f}%   (pctFmt = v x 100)")
            print(f"      6. PPTX        {_pct(frac):.4f}%   (deck _pct)")
            if frac < 0.02:
                print("      !! sub-2% weighted LTV — the payload is ~100x too "
                      "small for a mortgage book. Compare layer 2's scale against "
                      "layer 3: a POINTS classification on already-fractional "
                      "data divides twice.")
        print()

    payload = evolution.funded_cohort_progression(
        args.root, args.client, lens_filters=lens, vintage=args.vintage,
        grain=args.grain)
    print(f"  methodology={payload.get('methodologyVersion')} "
          f"membership={payload.get('membershipRule')} "
          f"formation={payload.get('formationDate')} "
          f"formationLoans={payload.get('formationLoanCount')}")
    for i, p in enumerate(payload.get("periods") or ()):
        v = (p.get("metrics") or {}).get("wa_ltv")
        print(f"   +{i} periods  {p.get('reporting_date')}  "
              f"surviving={p.get('survivingLoanCount')}  wa_ltv={v}  "
              f"React={'—' if v is None else f'{v * 100:.2f}%'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
