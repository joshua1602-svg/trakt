"""THE RECOVERY GATE — the offline evidence that would have stopped `fe7fea03`.

Every assertion in this package was run against BOTH builds before it was
committed, because an assertion that fails on both proves nothing about a
regression and an assertion that passes on both catches nothing at all. What
each one actually separates:

    assertion                                   ea8c65b   fe7fea03   meaning
    ----------------------------------------------------------------------------
    ITL3 surface stays available (3)            PASS      FAIL       G6 REGRESSION
    bridge and period change agree (1)          PASS      FAIL       G1 REGRESSION
    relative wording honours the gap policy (5) FAIL      FAIL       pre-existing
    zero comparable metrics is not ok (6)       FAIL      FAIL       pre-existing
    a distribution by balance measures money(3) FAIL      FAIL       pre-existing
    a real weighting qualifier still weights(1) FAIL      PASS       CANDIDATE GAIN
    portfolio_summary never raises (25)         PASS      PASS       NOT REPRODUCED

TWO CONSEQUENCES, BOTH OF WHICH CHANGE WHAT THE RECOVERY IS.

First, the gap-policy and zero-comparable-metrics failures are NOT what the
consolidation broke. Both builds answer a "month on month" question by spanning
212 days, and both publish "0 of 1 governed metrics could be compared" as a
successful answer. Those are standing defects that the target state rules out;
they are not restorations, and repairing them cannot be measured as removing a
regression.

Second, the `portfolio_summary` route exception is not reproduced by anything
here. Twenty-five governed geography shapes — region names, harmonised
taxonomy, ITL3 codes the ladder accepts and codes it rejects, postcodes that
resolve and postcodes that do not, no geography at all, and every asymmetric
pairing of an earlier snapshot against a later one — all answer. Whatever the
live book does to that route, this package cannot yet say.

WHY THE ITL3 FIXTURE IS THE SHAPE IT IS. `mi_geography` decides whether a column
carries geography by probing a head sample of its VALUES against a region
ladder, and returns true if any one of them reads as a place. Applied to a
column of ITL3 CODES that test is a coin toss on vocabulary: `TLC22` is rejected
and `TLD33` accepted, `TLK12` rejected and `TLK41` accepted. A demo book of
thirty-six loans drawn from recognised areas passes it every time; a live book
whose first rows happen to hold rejected codes loses its entire geography
surface. That is not a fixture gap that a bigger fixture would close — it is a
gate that has no business being value-dependent, and the assertions here say so
by holding the column fixed and moving only the codes inside it.
"""
