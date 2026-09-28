"""Tests for the interpretation/compiler boundary (mi_agent.interpretation_v2).

Isolated on purpose: nothing here imports a route, an executor or an engine, and
nothing here runs a plan. The boundary under test ends at the plan.
"""
