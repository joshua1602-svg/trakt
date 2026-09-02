"""The OCC configuration resolver.

Identifies client/portfolio/asset/regime, loads the active config packages,
applies the explicit precedence contract (system < regime < asset < client <
portfolio < approved run decisions) using the EXISTING
``config.system.config_resolver`` merge engine (per-key provenance), detects
conflicts and missing values, and persists one immutable
:class:`EffectiveConfiguration` before the Onboarding Agent runs.

Outcomes: READY / READY_WITH_WARNINGS / BLOCKED — operator-safe messages only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..annex2 import preflight as _preflight
from ..contracts import now_iso
from ..rules import RuleRecord, RuleStore
from ..stores import OpsStore, _write_json, _read_json
from . import contract as _contract
from .packages import (
    ASSET_MODEL,
    ConfigPackageStore,
    LAYER_ASSET,
    LAYER_REGIME,
    LAYER_SYSTEM,
)

READY = "READY"
READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
BLOCKED = "BLOCKED"

REPO = Path(__file__).resolve().parents[2]


@dataclass
class ResolutionOutcome:
    status: str
    effective: Optional[_contract.EffectiveConfiguration] = None
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    missing: List[Dict[str, Any]] = field(default_factory=list)   # preflight items


class EffectiveConfigResolver:
    def __init__(self, store: OpsStore, rules: RuleStore,
                 packages: Optional[ConfigPackageStore] = None,
                 client_config_path: Optional[Path] = None):
        self.store = store
        self.rules = rules
        self.packages = packages or ConfigPackageStore(store)
        self.client_config_path = Path(client_config_path
                                       or "config/client/config_client_ERM_UK.yaml")
        self._generated_cache: Dict[str, Path] = {}

    # ------------------------------------------------------------------ #
    def client_config_for(self, client_id: str) -> Path:
        """The client configuration in force for ``client_id``.

        A client that has been through Client Onboarding is governed by the
        configuration onboarding GENERATED for it — same format, same layer,
        same readers, but versioned and attributable. A client that has not is
        governed by the repository file exactly as before, so adopting a client
        is a decision, never a side effect of this code shipping.
        """
        if not client_id:
            return self.client_config_path
        cached = self._generated_cache.get(client_id)
        try:
            from ..onboarding.artefacts import client_config_rel
            from ..onboarding.store import OnboardingStore
            text = OnboardingStore(self.store).read_artefact(
                client_id, client_config_rel(client_id))
        except Exception:      # noqa: BLE001 — never break a delivery over this
            return cached or self.client_config_path
        if not text:
            return self.client_config_path
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if cached is not None and cached.name.startswith(digest):
            return cached
        import tempfile
        path = (Path(tempfile.mkdtemp(prefix="trakt-client-config-"))
                / f"{digest}-config_client_{client_id}.yaml")
        path.write_text(text, encoding="utf-8")
        self._generated_cache[client_id] = path
        return path

    # ------------------------------------------------------------------ #
    def resolve(self, *, client_id: str, portfolio_id: str,
                asset_type: str = "equity_release",
                outcome: str = "mi", reporting_period: str = "",
                workflow_id: str = "", created_by: str = "system",
                version: int = 1) -> ResolutionOutcome:
        blockers: List[str] = []
        warnings: List[str] = []
        conflicts: List[Dict[str, Any]] = []
        # The client layer: onboarding-generated where the client has been
        # onboarded, the repository file otherwise.
        client_config_path = self.client_config_for(client_id)

        # 1-4. Identify asset + applicable regime(s) via the support model.
        asset = ASSET_MODEL.get(asset_type)
        if asset is None:
            return ResolutionOutcome(
                status=BLOCKED,
                blockers=[f"The asset type '{asset_type}' is not configured "
                          "in Trakt. Ask your administrator."])
        regime_id = ""
        if outcome == "mi_annex2":
            if "ESMA_Annex2" not in asset["supports_regimes"]:
                return ResolutionOutcome(
                    status=BLOCKED,
                    blockers=[f"{asset['label']} portfolios do not support "
                              "this regulatory report."])
            regime_id = "ESMA_Annex2"

        # 5. Resolve the active governed packages (seeded from repo v1).
        pkg_system = self.packages.ensure_seeded(LAYER_SYSTEM)
        pkg_regime = self.packages.ensure_seeded(LAYER_REGIME)
        pkg_asset = self.packages.ensure_seeded(LAYER_ASSET)

        # 6-7. Scalar layers with the explicit precedence contract, using the
        # EXISTING merge engine (per-key provenance).
        import importlib.util
        import sys as _sys
        if "trakt_config_resolver" in _sys.modules:
            cr = _sys.modules["trakt_config_resolver"]
        else:
            spec = importlib.util.spec_from_file_location(
                "trakt_config_resolver",
                REPO / "config/system/config_resolver.py")
            cr = importlib.util.module_from_spec(spec)
            _sys.modules["trakt_config_resolver"] = cr
            spec.loader.exec_module(cr)

        client_rules = self._scoped(client_id, portfolio_id, workflow_id)
        client_overrides = {p["setting"]: p["value"] for r in
                            client_rules["client"] for p in [r.payload]
                            if p.get("setting")}
        portfolio_overrides = {p["setting"]: p["value"] for r in
                               client_rules["portfolio"] for p in [r.payload]
                               if p.get("setting")}
        run_overrides = {p["setting"]: p["value"] for r in
                         client_rules["run"] for p in [r.payload]
                         if p.get("setting")}

        asset_defaults = self._pkg_yaml(pkg_asset,
                                        "config/asset/product_defaults_ERM.yaml")
        resolved = cr.resolve_layers(
            context={"client": client_id, "portfolio": portfolio_id,
                     "asset": asset_type, "regime": regime_id or "none"},
            layers=[
                ("system", None),
                ("regime", None),
                ("asset", self._as_temp_yaml(asset_defaults, "asset")),
                ("client", client_config_path
                 if client_config_path.exists() else None),
            ],
            runtime_overrides=None)
        # Client-rule / portfolio / run overrides applied in precedence order
        # with explicit provenance (most specific last => wins).
        merged = resolved.resolved
        provenance = resolved.provenance
        overrides_log: List[Dict[str, Any]] = []
        for layer_name, ov, recs in (
                ("client", client_overrides, client_rules["client"]),
                ("portfolio", portfolio_overrides, client_rules["portfolio"]),
                ("run_decisions", run_overrides, client_rules["run"])):
            for key, value in ov.items():
                original = self._deep_get(merged, key)
                if original is not None and str(original) != str(value):
                    overrides_log.append(self._override_record(
                        key, original, value, layer_name, recs))
                    if layer_name == "portfolio":
                        conflicts.append({
                            "key": key, "client_value": original,
                            "portfolio_value": value,
                            "resolution": "portfolio override wins"})
                        warnings.append(
                            "This portfolio overrides a client-level setting "
                            f"({key.replace('_', ' ')}).")
                self._deep_set(merged, key, value)
                provenance[key] = layer_name

        # 8-9. Missing values / client completeness (operator-safe preflight).
        pf = _preflight.run_preflight(
            client_config_path=client_config_path,
            overrides={**client_overrides, **portfolio_overrides,
                       **run_overrides},
            reporting_period=reporting_period)
        missing = pf["blocked"]
        if outcome == "mi_annex2":
            blockers.extend(c["problem"] for c in missing)
        warnings.extend(c["problem"] for c in pf["needs_review"])

        # 10. Approved decision rule sets by kind (already scoped/approved).
        all_rules = (client_rules["global"] + client_rules["asset"]
                     + client_rules["client"] + client_rules["portfolio"]
                     + client_rules["file"] + client_rules["run"])
        mapping_rules = [self._rule_ref(r) for r in all_rules
                         if r.kind == "field_mapping"]
        alias_rules = [self._rule_ref(r) for r in all_rules
                       if r.kind == "alias"]
        enum_rules = [self._rule_ref(r) for r in all_rules
                      if r.kind == "enum"]

        decision_set_version = self._decision_set_version(all_rules)

        # 11-12. Build + persist the immutable contract.
        effective = _contract.build(
            client_id=client_id, portfolio_id=portfolio_id,
            asset_type=asset_type, regime_id=regime_id,
            regime_version=(f"v{pkg_regime['version']}" if regime_id else ""),
            reporting_date=reporting_period,
            system_config_version=f"v{pkg_system['version']}",
            regime_config_version=f"v{pkg_regime['version']}",
            asset_config_version=f"v{pkg_asset['version']}",
            client_config_version=self._file_version(client_config_path),
            portfolio_config_version=decision_set_version,
            decision_set_version=decision_set_version,
            layer_files={
                LAYER_SYSTEM: self.packages.file_hashes(LAYER_SYSTEM),
                LAYER_REGIME: self.packages.file_hashes(LAYER_REGIME),
                LAYER_ASSET: self.packages.file_hashes(LAYER_ASSET),
            },
            resolved_values=merged,
            value_provenance=provenance,
            overrides=overrides_log,
            mapping_rules=mapping_rules, alias_rules=alias_rules,
            enum_rules=enum_rules,
            validation_policy={"source": "system package",
                               "version": f"v{pkg_system['version']}"},
            file_recognition_policy={"source": "system package "
                                               "(onboarding modes)"},
            schema_recognition_policy={"source": "system package "
                                                 "(registry + aliases)"},
            canonical_registry_reference="config/system/fields_registry.yaml"
                                         f"@v{pkg_system['version']}",
            source_provenance={
                "system": f"config package v{pkg_system['version']}",
                "regime": f"config package v{pkg_regime['version']}",
                "asset": f"config package v{pkg_asset['version']}",
                "client": ("Client Onboarding (generated client "
                           "configuration)"
                           if client_config_path != self.client_config_path
                           else str(self.client_config_path)),
                "portfolio": "OCC rule store (portfolio scope)",
                "run_decisions": "OCC rule store (run scope + approved "
                                 "workflow decisions)",
            },
            conflicts=conflicts, warnings=warnings, blockers=blockers,
            created_by=created_by, version=version)

        if blockers:
            outcome_obj = ResolutionOutcome(status=BLOCKED, effective=effective,
                                            blockers=blockers,
                                            warnings=warnings,
                                            conflicts=conflicts,
                                            missing=missing)
        elif warnings:
            outcome_obj = ResolutionOutcome(status=READY_WITH_WARNINGS,
                                            effective=effective,
                                            warnings=warnings,
                                            conflicts=conflicts)
        else:
            outcome_obj = ResolutionOutcome(status=READY, effective=effective)

        if workflow_id:
            self.persist(client_id, workflow_id, effective,
                         status=outcome_obj.status)
        return outcome_obj

    # ------------------------------------------------------------------ #
    def persist(self, client_id: str, workflow_id: str,
                effective: _contract.EffectiveConfiguration,
                *, status: str) -> str:
        base = (f"blob://{self.store.layout.container}/{client_id}/"
                f"workflow-runs/{workflow_id}/effective-config")
        uri = f"{base}/{effective.version:04d}.json"
        _write_json(self.store.storage, uri,
                    {"status": status, **effective.to_dict()})
        _write_json(self.store.storage, f"{base}/current.json",
                    {"version": effective.version,
                     "effective_config_id": effective.effective_config_id,
                     "content_hash": effective.content_hash,
                     "status": status})
        return uri

    def load(self, client_id: str, workflow_id: str,
             version: Optional[int] = None
             ) -> Optional[_contract.EffectiveConfiguration]:
        base = (f"blob://{self.store.layout.container}/{client_id}/"
                f"workflow-runs/{workflow_id}/effective-config")
        if version is None:
            cur = _read_json(self.store.storage, f"{base}/current.json")
            if not cur:
                return None
            version = int(cur["version"])
        doc = _read_json(self.store.storage, f"{base}/{version:04d}.json")
        return (_contract.EffectiveConfiguration.from_dict(doc)
                if doc else None)

    def materialise_snapshot(self, effective: _contract.EffectiveConfiguration,
                             dest: Path) -> Dict[str, str]:
        """Write hash-verified snapshots of every pinned layer file + the
        effective client configuration; returns the paths the narrowed agent
        consumes (registry, aliases_dir, client_config, ...)."""
        dest.mkdir(parents=True, exist_ok=True)
        for layer, ver_key in ((LAYER_SYSTEM, "system_config_version"),
                               (LAYER_REGIME, "regime_config_version"),
                               (LAYER_ASSET, "asset_config_version")):
            version = int(getattr(effective, ver_key).lstrip("v"))
            self.packages.materialise(layer, version, dest)
        client_cfg = dest / "config/client/effective_client_config.yaml"
        client_cfg.parent.mkdir(parents=True, exist_ok=True)
        client_cfg.write_text(
            "# GENERATED effective client configuration (immutable snapshot)\n"
            + yaml.safe_dump(effective.resolved_values, sort_keys=False),
            encoding="utf-8")
        return {
            "registry": str(dest / "config/system/fields_registry.yaml"),
            "aliases_dir": str(dest / "config/system"),
            "asset_config": str(dest /
                                "config/asset/product_defaults_ERM.yaml"),
            "client_config": str(client_cfg),
            "snapshot_root": str(dest),
        }

    # ------------------------------------------------------------------ #
    def _scoped(self, client_id: str, portfolio_id: str,
                workflow_id: str) -> Dict[str, List[RuleRecord]]:
        out: Dict[str, List[RuleRecord]] = {s: [] for s in
                                            ("file", "run", "portfolio",
                                             "client", "asset", "global")}
        for r in self.rules.list_current(client_id):
            if r.status != "active":
                continue
            if r.scope == "portfolio" and r.portfolio_id != portfolio_id:
                continue
            if r.scope == "run" and r.run_ref != workflow_id:
                continue
            if r.scope in out:
                out[r.scope].append(r)
        for r in self.rules.list_current(None):
            if r.status == "active" and r.scope in ("global", "asset"):
                out[r.scope].append(r)
        return out

    @staticmethod
    def _rule_ref(r: RuleRecord) -> Dict[str, Any]:
        return {"rule_id": r.rule_id, "version": r.version, "kind": r.kind,
                "scope": r.scope, "payload": r.payload,
                "approved_by": r.approved_by, "approved_at": r.approved_at}

    @staticmethod
    def _override_record(key, original, value, layer, recs) -> Dict[str, Any]:
        rec = next((r for r in recs
                    if (r.payload or {}).get("setting") == key), None)
        return {"key": key, "original_value": original,
                "overridden_value": value, "layer": layer,
                "override_reason": (rec.reason if rec else ""),
                "approved_by": (rec.approved_by if rec else ""),
                "approved_at": (rec.approved_at if rec else ""),
                "scope": (rec.scope if rec else layer),
                "rule_id": (rec.rule_id if rec else "")}

    def _decision_set_version(self, rules: List[RuleRecord]) -> str:
        blob = json.dumps(sorted(f"{r.rule_id}:{r.version}" for r in rules))
        return "d" + hashlib.sha256(blob.encode()).hexdigest()[:12]

    def _pkg_yaml(self, pkg: Dict[str, Any], rel: str) -> Dict[str, Any]:
        f = (pkg.get("files") or {}).get(rel)
        return yaml.safe_load(f["content"]) or {} if f else {}

    _tmp_counter = 0

    def _as_temp_yaml(self, data: Dict[str, Any], name: str) -> Optional[Path]:
        if not data:
            return None
        import tempfile
        p = Path(tempfile.mkdtemp(prefix=f"ecfg-{name}-")) / f"{name}.yaml"
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return p

    @staticmethod
    def _file_version(path: Path) -> str:
        try:
            text = Path(path).read_text(encoding="utf-8")
            return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:12]
        except OSError:
            return "absent"

    @staticmethod
    def _deep_get(node: Dict[str, Any], key: str) -> Any:
        if key in node:
            return node[key]
        for v in node.values():
            if isinstance(v, dict):
                hit = EffectiveConfigResolver._deep_get(v, key)
                if hit is not None:
                    return hit
        return None

    @staticmethod
    def _deep_set(node: Dict[str, Any], key: str, value: Any) -> None:
        if key in node:
            node[key] = value
            return
        for v in node.values():
            if isinstance(v, dict) and \
                    EffectiveConfigResolver._deep_get(v, key) is not None:
                EffectiveConfigResolver._deep_set(v, key, value)
                return
        node[key] = value
