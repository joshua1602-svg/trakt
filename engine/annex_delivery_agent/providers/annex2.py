"""
engine.annex_delivery_agent.providers.annex2
============================================

ESMA Annex 2 (non-ABCP underlying exposures, ``auth.099.001.04``).

This provider **wraps** the proven route. It does not reimplement any part of
it, and it changes no calculation, rule or mapping:

===================  ==================================================================
Projector            ``engine/gate_4_projection/regime_projector.py``
Delivery normaliser  ``engine/gate_4b_delivery/annex2_delivery_normalizer.py``
XML builder          ``engine/gate_5_delivery/xml_builder_annex2.py``
Schema               ``config/system/DRAFT1auth.099.001.04_1.3.0.xsd``
Validation           ``lxml.etree.XMLSchema`` over that XSD
===================  ==================================================================

Two deliberate choices about *how* it wraps them:

* The normaliser and builder are called as **library functions**
  (``normalize_delivery``, ``build_annex2_tree``), not as subprocesses. Their
  ``main()`` entry points add argument parsing and a ``sys.exit(2)`` hard gate;
  the functions beneath are pure and are exactly what the CLI runs. Calling them
  directly keeps the business logic identical while letting the agent hold the
  blocking decision, the timings and the evidence.

* Everything the builder invents is *observed*, never prevented. See
  :mod:`engine.annex_delivery_agent.instrumentation`.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

def _derived_contract_path() -> str:
    """The effective Annex 2 contract, materialised for a path-taking caller."""
    from engine.regime_contract.annex2_contract import materialised_contract_path
    return materialised_contract_path()


from ..contracts import (
    AnnexDefinition, ArtefactRef, DeliveryRequest, SchemaValidationResult,
    config_fingerprints, file_sha256,
)
from ..evidence import (
    SEVERITY_INFO, SEVERITY_WARNING, AutomaticFill, CoercionCollector, Omission,
    StructuralAssumption, TransformationEvidence, UnsupportedField,
)
from ..instrumentation import (
    analyse_branch_coverage, count_nodata_in_frame, count_paths, describe_path,
    discover_nodata_paths, mapped_nodata_paths, observe_branch_drops,
    observe_coercions, owning_element, total_leaves,
)
from ..persistence import StagedFile
from ..preflight import (
    PreflightReport, require_file, require_non_empty_input, require_reporting_date,
)
from ..validators.xsd import LxmlXsdValidator
from .base import AnnexProvider, BuildOutcome, NormalisationOutcome, ResolvedInput

_REPO_ROOT = Path(__file__).resolve().parents[3]

ANNEX_ID = "ESMA_Annex2"
ANNEX_NAME = "Annex 2"
REPORT_VERSION = "auth.099.001.04_1.3.0"

#: The 20-character LEI shape ESMA requires (18 alphanumeric + 2 check digits).
_LEI_RE = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")
#: RREL1 embeds the reporting entity's LEI: ``<LEI>N<year><sequence>``.
_SECURITISATION_ID_RE = re.compile(r"^([A-Z0-9]{18}[0-9]{2})N\d{4}\d{2}$")

#: The XML element that repeats once per exposure.
RECORD_ANCHOR = "UndrlygXpsrRcrd"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Annex2Paths:
    """Every file the Annex 2 route reads.

    Defaults are the exact paths the proven ``engine/orchestrator/trakt_run.py``
    regulatory route uses — discovered from the repository, not invented, and
    not copied to an alternative location.
    """

    #: Empty means "derive it". The Annex 2 contract comes from the field
    #: universe, the registry, the workbook and the XSD; there is no
    #: delivery-rules file. Resolved lazily so importing this module is cheap.
    delivery_rules: str = ""
    field_universe: str = str(_REPO_ROOT / "config" / "regime" / "annex2_field_universe.yaml")
    code_order: str = str(_REPO_ROOT / "config" / "system" / "esma_code_order.yaml")
    enum_mapping: str = str(_REPO_ROOT / "config" / "system" / "enum_mapping.yaml")
    mapping_workbook: str = str(
        _REPO_ROOT / "DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report_Version_1.3.1.xlsx")
    mapping_sheet: str = "DRAFT1auth.099.001.04"
    xsd: str = str(_REPO_ROOT / "config" / "system" / "DRAFT1auth.099.001.04_1.3.0.xsd")
    client_config: Optional[str] = None

    def with_overrides(self, options: Mapping[str, Any],
                       client_config: Optional[str]) -> "Annex2Paths":
        return Annex2Paths(
            delivery_rules=str(options.get("delivery_rules") or self.delivery_rules
                               or _derived_contract_path()),
            field_universe=str(options.get("field_universe") or self.field_universe),
            code_order=str(options.get("code_order") or self.code_order),
            enum_mapping=str(options.get("enum_mapping") or self.enum_mapping),
            mapping_workbook=str(options.get("mapping_workbook") or self.mapping_workbook),
            mapping_sheet=str(options.get("mapping_sheet") or self.mapping_sheet),
            xsd=str(options.get("xsd") or self.xsd),
            client_config=str(client_config) if client_config else self.client_config,
        )

    def fingerprint_map(self) -> Dict[str, str]:
        paths = {
            "annex2_contract": self.delivery_rules,
            "annex2_field_universe": self.field_universe,
            "esma_code_order": self.code_order,
            "enum_mapping": self.enum_mapping,
            "annex2_mapping_workbook": self.mapping_workbook,
            "annex2_xsd": self.xsd,
        }
        if self.client_config:
            paths["client_config"] = self.client_config
        return paths


ANNEX2_DEFINITION = AnnexDefinition(
    annex_id=ANNEX_ID,
    annex_name="ESMA Annex 2 — Underlying exposures (residential real estate)",
    jurisdiction="EU/UK",
    regulatory_regime="Securitisation Regulation disclosure RTS/ITS",
    report_version=REPORT_VERSION,
    artefact_type="xml",
    authoritative_projector="engine/gate_4_projection/regime_projector.py",
    normaliser="engine/gate_4b_delivery/annex2_delivery_normalizer.py::normalize_delivery",
    builder="engine/gate_5_delivery/xml_builder_annex2.py::build_annex2_tree",
    validator="lxml.etree.XMLSchema",
    schema_reference="config/system/DRAFT1auth.099.001.04_1.3.0.xsd",
    field_universe_reference="config/regime/annex2_field_universe.yaml",
    rules_reference="engine.regime_contract.annex2_contract (derived)",
    code_order_reference="config/system/esma_code_order.yaml",
    configuration_references=(
        "config/system/enum_mapping.yaml",
        "DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report_Version_1.3.1.xlsx",
    ),
    supported_input_contracts=("*_ESMA_Annex2_projected.csv",),
    output_contract="urn:esma:xsd:DRAFT1auth.099.001.04 Document",
    publication_target="regulatory_delivery/esma_annex2",
)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class Annex2Provider(AnnexProvider):
    """Delivery behaviour for ESMA Annex 2."""

    definition = ANNEX2_DEFINITION

    def __init__(self, paths: Optional[Annex2Paths] = None) -> None:
        self._base_paths = paths or Annex2Paths()
        self.paths: Annex2Paths = self._base_paths
        # Per-run state, populated as the lifecycle proceeds.
        self._specs_by_code: Dict[str, List[Any]] = {}
        self._namespace: str = ""
        self._coercions = CoercionCollector()
        self._drop_counter: Counter = Counter()
        self._rules: Dict[str, Any] = {}

    # -- configuration ------------------------------------------------------ #
    def check_configuration(self, request: DeliveryRequest,
                            report: PreflightReport) -> Mapping[str, str]:
        self.paths = self._base_paths.with_overrides(
            request.options or {}, request.client_config_reference)

        require_file(report, check="annex2_contract", path=self.paths.delivery_rules,
                     what="the Annex 2 delivery rules", annex_name=ANNEX_NAME)
        require_file(report, check="annex2_field_universe", path=self.paths.field_universe,
                     what="the Annex 2 field universe", annex_name=ANNEX_NAME)
        require_file(report, check="esma_code_order", path=self.paths.code_order,
                     what="the ESMA field-order definition", annex_name=ANNEX_NAME)
        require_file(report, check="annex2_mapping_workbook", path=self.paths.mapping_workbook,
                     what="the authoritative Annex 2 mapping workbook (version 1.3.1)",
                     annex_name=ANNEX_NAME)
        require_file(report, check="annex2_xsd", path=self.paths.xsd,
                     what="the official Annex 2 XML schema (XSD)", annex_name=ANNEX_NAME)
        if self.paths.client_config:
            require_file(report, check="client_config", path=self.paths.client_config,
                         what="the client's Annex 2 configuration", annex_name=ANNEX_NAME)

        return config_fingerprints(self.paths.fingerprint_map())

    def check_applicability(self, request: DeliveryRequest,
                            report: PreflightReport) -> bool:
        """Annex 2 applies unless the client's configuration says it does not."""
        if not self.paths.client_config:
            return True
        config = _load_yaml(Path(self.paths.client_config))
        if config is None:
            return True
        supported = config.get("supported_regimes")
        if isinstance(supported, (list, tuple)) and supported:
            if ANNEX_ID not in {str(s).strip() for s in supported}:
                report.block(
                    "report_applicability",
                    f"{ANNEX_NAME} is not enabled for this client. The client's "
                    f"configuration lists "
                    f"{', '.join(str(s) for s in supported)} as the reports it may submit.",
                    remedy=f"Add {ANNEX_ID} to the client's supported reports before "
                           f"preparing this submission.",
                    supported_regimes=[str(s) for s in supported])
                return False
        return True

    # -- input -------------------------------------------------------------- #
    def resolve_input(self, request: DeliveryRequest,
                      report: PreflightReport) -> Optional[ResolvedInput]:
        path = require_file(report, check="projected_input", path=request.projected_input,
                            what="the projected Annex 2 data for this reporting period",
                            annex_name=ANNEX_NAME,
                            remedy="Run the projection stage for this portfolio and "
                                   "reporting date, then prepare the report again.")
        if path is None:
            return None

        header, row_count = _scan_csv(path)
        if header is None:
            report.block("projected_input_shape",
                         f"{ANNEX_NAME} cannot be prepared because the projected data "
                         f"file is empty or has no column headings.",
                         remedy="Re-run the projection stage for this reporting date.",
                         path=str(path))
            return None

        return ResolvedInput(
            path=path,
            row_count=row_count,
            field_count=len(header),
            field_codes=tuple(header),
            content_hash=file_sha256(path),
        )

    # -- preflight ---------------------------------------------------------- #
    def preflight(self, request: DeliveryRequest, resolved: Optional[ResolvedInput],
                  report: PreflightReport) -> None:
        require_reporting_date(report, reporting_date=request.reporting_date,
                               annex_name=ANNEX_NAME)
        if resolved is None:
            return

        require_non_empty_input(report, row_count=resolved.row_count, annex_name=ANNEX_NAME)
        self._check_input_schema(resolved, report)
        self._check_reporting_entity_lei(resolved, report)

    def _check_input_schema(self, resolved: ResolvedInput, report: PreflightReport) -> None:
        """The projected columns must be Annex 2 codes the rules know about."""
        rules = self._load_rules()
        required = {code for code, rule in (rules.get("field_rules") or {}).items()
                    if isinstance(rule, dict) and rule.get("enforce_presence")}
        present = set(resolved.field_codes)
        missing = sorted(required - present)
        if missing:
            report.block(
                "input_schema_compatibility",
                f"{ANNEX_NAME} cannot be prepared because the projected data is "
                f"missing {len(missing)} field(s) that Annex 2 requires: "
                f"{', '.join(missing[:8])}"
                f"{'…' if len(missing) > 8 else ''}.",
                remedy="Re-run the projection stage with the current Annex 2 field "
                       "configuration, or correct the client's field mapping.",
                missing_codes=missing)

        annex_shaped = [c for c in resolved.field_codes if re.match(r"^[A-Z]{4}\d+$", str(c))]
        if not annex_shaped:
            report.block(
                "input_schema_compatibility",
                f"{ANNEX_NAME} cannot be prepared because the supplied file does not "
                f"look like projected Annex 2 data — none of its columns are Annex 2 "
                f"field codes.",
                remedy="Supply the output of the Annex 2 projection stage "
                       "(*_ESMA_Annex2_projected.csv).")

    def _check_reporting_entity_lei(self, resolved: ResolvedInput,
                                    report: PreflightReport) -> None:
        """The submission must carry a usable reporting-entity LEI.

        Annex 2 carries it two ways: RREL1 embeds it in the securitisation
        identifier, and RREL83 holds the originator's LEI. Either satisfies the
        check; neither, or a malformed one, blocks the run before any file is
        built.
        """
        import pandas as pd

        candidates = [c for c in ("RREL1", "RREL83") if c in resolved.field_codes]
        if not candidates:
            report.block(
                "reporting_entity_lei",
                f"{ANNEX_NAME} cannot be prepared because the reporting entity LEI "
                f"has not been configured for this client.",
                remedy="Configure the reporting entity's 20-character LEI in the "
                       "client's Annex 2 configuration and re-run the projection.",
                looked_for=["RREL1", "RREL83"])
            return

        frame = pd.read_csv(resolved.path, dtype=str, usecols=candidates,
                            nrows=_LEI_PROBE_ROWS).fillna("")
        for column in candidates:
            for raw in frame[column].tolist():
                value = str(raw).strip().upper()
                if not value:
                    continue
                if _LEI_RE.match(value):
                    return
                match = _SECURITISATION_ID_RE.match(value)
                if match and _LEI_RE.match(match.group(1)):
                    return

        report.block(
            "reporting_entity_lei",
            f"{ANNEX_NAME} cannot be prepared because the reporting entity LEI has "
            f"not been configured for this client — the projected data carries no "
            f"valid 20-character LEI in {' or '.join(candidates)}.",
            remedy="Configure the reporting entity's LEI for this client and re-run "
                   "the projection stage.",
            looked_in=candidates)

    # -- normalisation ------------------------------------------------------ #
    def normalise(self, request: DeliveryRequest, resolved: ResolvedInput,
                  work_dir: Path) -> NormalisationOutcome:
        """Run the frozen Gate 4b normaliser and persist its outputs.

        Calls ``normalize_delivery`` — the pure function the Gate 4b CLI wraps —
        so the normalisation rules are byte-for-byte the proven ones. The CLI's
        ``sys.exit(2)`` hard gate is reproduced by the agent as a blocking status,
        not swallowed: ``blocking_issue_count > 0`` stops the lifecycle before the
        builder runs.
        """
        import pandas as pd

        from engine.gate_4b_delivery.annex2_delivery_normalizer import normalize_delivery

        rules = self._load_rules()
        frame = pd.read_csv(resolved.path, dtype=str).fillna("")
        out_df, issues, summary = normalize_delivery(frame, rules)

        base = resolved.path.stem
        if base.endswith("_projected"):
            base = base[: -len("_projected")]

        delivery_ready = work_dir / f"{base}_delivery_ready.csv"
        with StagedFile(delivery_ready) as staged:
            out_df.to_csv(staged.temp_path, index=False)

        issue_rows = [i.as_dict() for i in issues]
        issues_csv = work_dir / f"{base}_delivery_issues.csv"
        with StagedFile(issues_csv) as staged:
            pd.DataFrame(issue_rows).to_csv(staged.temp_path, index=False)

        report_json = work_dir / f"{base}_delivery_report.json"
        import json
        with StagedFile(report_json) as staged:
            staged.temp_path.write_text(
                json.dumps({"input": str(resolved.path),
                            "rules": self.paths.delivery_rules,
                            **summary}, indent=2, default=str),
                encoding="utf-8")

        blocking = int((summary.get("preflight") or {}).get("blocking_errors") or 0)
        return NormalisationOutcome(
            artefact=ArtefactRef.of_file("delivery_ready_csv", delivery_ready),
            row_count=int(len(out_df)),
            field_count=int(len(out_df.columns)),
            issues=[r for r in issue_rows if r.get("severity") == "error"][:200],
            blocking_issue_count=blocking,
            summary=dict(summary),
            side_artefacts=[
                ArtefactRef.of_file("delivery_issues_csv", issues_csv),
                ArtefactRef.of_file("delivery_report_json", report_json),
            ],
            frame=out_df,
        )

    # -- build -------------------------------------------------------------- #
    def build(self, request: DeliveryRequest, normalised: NormalisationOutcome,
              work_dir: Path) -> BuildOutcome:
        """Run the frozen Gate 5 builder under observation."""
        import pandas as pd
        from lxml import etree

        from engine.gate_5_delivery import xml_builder_annex2 as builder

        frame = normalised.frame
        if frame is None:  # restart path: the delivery-ready CSV is the contract
            frame = pd.read_csv(normalised.artefact.path, dtype=str)

        options = dict(request.options or {})
        currency = str(options.get("currency") or self._client_currency() or "GBP")
        performance_mode = str(options.get("performance_mode") or "PRF").upper()

        code_order = builder.load_code_order(self.paths.code_order)
        specs_by_code = _load_mapping_specs_cached(
            builder, self.paths.mapping_workbook, self.paths.mapping_sheet,
            performance_mode)
        namespace = builder.get_namespace_from_xsd(self.paths.xsd)

        self._specs_by_code = specs_by_code
        self._namespace = namespace
        self._coercions = CoercionCollector()
        self._drop_counter = Counter()

        with observe_coercions(builder, "_coerce_record_value_for_branch", self._coercions), \
                observe_branch_drops(builder, "select_specs_for_value", self._drop_counter):
            root = builder.build_annex2_tree(
                frame, code_order, specs_by_code, namespace, currency, self.paths.xsd)

        artefact_path = work_dir / f"{_artefact_stem(request)}.xml"
        tree = etree.ElementTree(root)
        with StagedFile(artefact_path) as staged:
            tree.write(str(staged.temp_path), pretty_print=True,
                       xml_declaration=True, encoding="UTF-8")

        record_count = len(root.findall(f".//{{{namespace}}}{RECORD_ANCHOR}"))
        delivered = sum(1 for c in frame.columns if str(c) in specs_by_code)

        return BuildOutcome(
            artefact=ArtefactRef.of_file("annex2_xml", artefact_path),
            record_count=record_count,
            field_count=delivered,
            parsed=root,
            detail={
                "namespace": namespace,
                "currency": currency,
                "performance_mode": performance_mode,
                "mapping_sheet": self.paths.mapping_sheet,
                "mapped_code_count": len(specs_by_code),
            },
        )

    # -- validation --------------------------------------------------------- #
    def validate(self, request: DeliveryRequest,
                 built: BuildOutcome) -> SchemaValidationResult:
        return LxmlXsdValidator().validate(
            Path(built.artefact.path), schema_path=Path(self.paths.xsd),
            parsed=built.parsed)

    # -- evidence ----------------------------------------------------------- #
    def inspect(self, request: DeliveryRequest, resolved: ResolvedInput,
                normalised: NormalisationOutcome,
                built: BuildOutcome) -> TransformationEvidence:
        """Itemise what the builder filled, coerced, dropped or assumed."""
        import pandas as pd
        from engine.gate_5_delivery import xml_builder_annex2 as builder

        evidence = TransformationEvidence()
        frame = normalised.frame
        if frame is None:
            frame = pd.read_csv(normalised.artefact.path, dtype=str)
        frame = frame.fillna("")

        root = built.parsed
        namespace = self._namespace or str(built.detail.get("namespace") or "")
        specs_by_code = self._specs_by_code

        if root is not None and namespace and specs_by_code:
            self._collect_automatic_fills(evidence, root, frame, namespace, specs_by_code)
            self._collect_omissions(evidence, frame, specs_by_code, builder)

        evidence.coercions.extend(self._coercions.as_evidence(reason_for=_COERCION_REASONS))
        self._collect_unsupported(evidence, frame, specs_by_code)
        self._collect_structural_assumptions(evidence, frame, built)

        dropped_observed = int(self._drop_counter.get("dropped_values", 0))
        if dropped_observed and not evidence.omissions:  # pragma: no cover - defensive
            evidence.notes.append(
                f"{dropped_observed} value(s) were observed being dropped at branch "
                f"selection but could not be attributed to a specific field.")
        return evidence

    def _collect_automatic_fills(self, evidence: TransformationEvidence, root: Any,
                                 frame: Any, namespace: str,
                                 specs_by_code: Dict[str, List[Any]]) -> None:
        """Attribute every no-data leaf to the input data or to the builder."""
        xml_total = total_leaves(root, ns=namespace)
        csv_total, csv_per_field = count_nodata_in_frame(frame)

        paths = discover_nodata_paths(root, ns=namespace, record_anchor=RECORD_ANCHOR)
        counts = count_paths(root, paths, ns=namespace)
        sourced_paths = mapped_nodata_paths(specs_by_code, frame.columns)

        attributed = 0
        automatic_by_location: Dict[Tuple[str, str, str], int] = {}
        for path, count in counts.items():
            attributed += count
            if path in sourced_paths:
                continue
            location = describe_path(path, record_anchor=RECORD_ANCHOR)
            key = (owning_element(path), location, _fill_reason(path))
            automatic_by_location[key] = automatic_by_location.get(key, 0) + count

        for (owner, location, reason), count in sorted(automatic_by_location.items()):
            evidence.automatic_fills.append(AutomaticFill(
                field_code=owner,
                xml_path=location,
                records_affected=count,
                original_state="not supplied by the prepared data",
                generated_value="ND5",
                reason=reason,
                severity=SEVERITY_WARNING,
                review_required=True,
                detection="structural"))

        automatic_total = sum(automatic_by_location.values())
        evidence.nd_reconciliation = {
            "no_data_values_in_prepared_data": csv_total,
            "no_data_values_in_report": xml_total,
            "difference_added_by_builder": xml_total - csv_total,
            "attributed_by_structure": attributed,
            "attributed_as_automatic": automatic_total,
            "attributed_as_sourced": attributed - automatic_total,
            "unattributed": xml_total - attributed,
            "per_field_in_prepared_data": dict(sorted(csv_per_field.items())),
        }
        if xml_total != attributed:
            evidence.notes.append(
                f"{xml_total - attributed} no-data value(s) in the report sit at "
                f"element paths that were not seen in the sampled records; they are "
                f"counted in the totals but not attributed to a named location.")

    def _collect_omissions(self, evidence: TransformationEvidence, frame: Any,
                           specs_by_code: Dict[str, List[Any]], builder: Any) -> None:
        """Fields that carried data but could not be placed in the artefact."""
        coverage = analyse_branch_coverage(
            frame, specs_by_code, builder.select_specs_for_value,
            record_anchor=RECORD_ANCHOR)
        for code, detail in sorted(coverage.items()):
            evidence.omissions.append(Omission(
                field_code=code,
                records_affected=int(detail["records_affected"]),
                original_value_sample=tuple(detail["samples"]),
                reason=("the mapping workbook offers no non-no-data branch for this "
                        "value, and the field is optional in the schema, so the "
                        "builder omitted it"),
                optional=True,
                severity=SEVERITY_WARNING,
                review_required=True))

        unmapped = [str(c) for c in frame.columns if str(c) not in specs_by_code]
        annex_shaped = [c for c in unmapped if re.match(r"^[A-Z]{4}\d+$", c)]
        for code in sorted(annex_shaped):
            populated = int((frame[code].astype(str).str.strip() != "").sum())
            if not populated:
                continue
            evidence.omissions.append(Omission(
                field_code=code,
                records_affected=populated,
                original_value_sample=tuple(
                    frame[code].astype(str).str.strip().replace("", None).dropna()
                    .unique()[:5].tolist()),
                reason=("this field has no mapping in the authoritative Annex 2 "
                        "workbook for the selected template, so it cannot be placed "
                        "in the report"),
                optional=True,
                severity=SEVERITY_WARNING,
                review_required=True))

    def _collect_unsupported(self, evidence: TransformationEvidence, frame: Any,
                             specs_by_code: Dict[str, List[Any]]) -> None:
        """Universe fields this route cannot emit at all."""
        universe = _load_yaml(Path(self.paths.field_universe)) or {}
        fields = universe.get("fields")
        if not isinstance(fields, dict):
            return
        present = {str(c) for c in frame.columns}
        for code, meta in sorted(fields.items()):
            code = str(code)
            if code in present:
                continue
            mandatory = not bool((meta or {}).get("nd5_allowed", True))
            if code in specs_by_code:
                reason = ("the report has a place for this field but the prepared "
                          "data does not carry it")
            else:
                reason = ("this field is in the Annex 2 universe but has no mapping "
                          "in the authoritative workbook for this template")
            evidence.unsupported_fields.append(UnsupportedField(
                field_code=code,
                field_name=str((meta or {}).get("field_name") or ""),
                reason=reason,
                mandatory=mandatory,
                severity=SEVERITY_WARNING if mandatory else SEVERITY_INFO,
                review_required=mandatory))

    def _collect_structural_assumptions(self, evidence: TransformationEvidence,
                                        frame: Any, built: BuildOutcome) -> None:
        rows = int(len(frame))
        evidence.structural_assumptions.append(StructuralAssumption(
            name="one_record_per_row",
            description=(f"The report contains exactly one underlying-exposure record "
                         f"for each of the {rows:,} rows of prepared data. If your "
                         f"source represents a single exposure across several rows, "
                         f"those rows appear as separate exposures."),
            records_affected=rows,
            severity=SEVERITY_INFO,
            review_required=False))

        collateral_codes = [str(c) for c in frame.columns if str(c).startswith("RREC")]
        if collateral_codes:
            evidence.structural_assumptions.append(StructuralAssumption(
                name="single_collateral_per_exposure",
                description=(f"Collateral is reported as one collateral item per "
                             f"exposure ({len(collateral_codes)} collateral fields, one "
                             f"value each). Where an exposure is secured on more than "
                             f"one property, only the values present in these columns "
                             f"are reported."),
                records_affected=rows,
                severity=SEVERITY_WARNING,
                review_required=True))

    # -- publication -------------------------------------------------------- #
    def publication_metadata(self, request: DeliveryRequest, resolved: ResolvedInput,
                             normalised: NormalisationOutcome, built: BuildOutcome,
                             validation: SchemaValidationResult) -> Dict[str, Any]:
        return {
            "annex_id": ANNEX_ID,
            "report_version": REPORT_VERSION,
            "namespace": built.detail.get("namespace"),
            "currency": built.detail.get("currency"),
            "performance_mode": built.detail.get("performance_mode"),
            "mapping_workbook": self.paths.mapping_workbook,
            "mapping_sheet": self.paths.mapping_sheet,
            "schema": self.paths.xsd,
            "schema_sha256": validation.schema_sha256,
            "projected_dataset": resolved.to_dict(),
            "normalised_dataset": normalised.artefact.to_dict(),
            "artefact": built.artefact.to_dict(),
            "record_count": built.record_count,
            "delivered_field_count": built.field_count,
        }

    # -- descriptive -------------------------------------------------------- #
    def component_versions(self) -> Mapping[str, str]:
        modules = {
            "regime_projector": _REPO_ROOT / "engine/gate_4_projection/regime_projector.py",
            "annex2_delivery_normalizer":
                _REPO_ROOT / "engine/gate_4b_delivery/annex2_delivery_normalizer.py",
            "xml_builder_annex2": _REPO_ROOT / "engine/gate_5_delivery/xml_builder_annex2.py",
            "annex2_provider": Path(__file__),
        }
        return {name: (file_sha256(p) if p.is_file() else "missing")
                for name, p in modules.items()}

    # -- internals ---------------------------------------------------------- #
    def _load_rules(self) -> Dict[str, Any]:
        if not self._rules:
            self._rules = _load_yaml(Path(self.paths.delivery_rules)) or {}
        return self._rules

    def _client_currency(self) -> Optional[str]:
        if not self.paths.client_config:
            return None
        config = _load_yaml(Path(self.paths.client_config)) or {}
        portfolio = config.get("portfolio")
        if isinstance(portfolio, dict):
            currency = portfolio.get("base_currency")
            if currency:
                return str(currency)
        return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: Rows read when probing for a reporting-entity LEI. The identifier is a
#: submission-level constant, so the first rows settle it without loading an
#: 11k-row frame during preflight.
_LEI_PROBE_ROWS = 200

_COERCION_REASONS = {
    "RREL12": ("The Annex 2 schema maps this field to a four-digit year. A value "
               "that is not a year cannot be placed in that branch, so it is "
               "reported as no-data rather than replaced with a substitute year "
               "(builder rule: _coerce_record_value_for_branch)."),
}


#: Parsed mapping workbooks, keyed by (path, mtime, size, sheet, performance mode).
#: Reading the 9.6 MB workbook costs several seconds and dominates a small run;
#: it is immutable input, so parsing it once per process is safe. Each caller
#: gets its own dict and lists so no run can affect another's specs.
_SPECS_CACHE: Dict[Tuple[str, int, int, str, str], Dict[str, List[Any]]] = {}


def _load_mapping_specs_cached(builder: Any, workbook: str, sheet: str,
                               performance_mode: str) -> Dict[str, List[Any]]:
    path = Path(workbook)
    stat = path.stat()
    key = (str(path), int(stat.st_mtime), int(stat.st_size), sheet, performance_mode)
    cached = _SPECS_CACHE.get(key)
    if cached is None:
        cached = builder.load_mapping_specs(workbook, sheet, performance_mode)
        _SPECS_CACHE[key] = cached
    return {code: list(specs) for code, specs in cached.items()}


def _fill_reason(path: Tuple[str, ...]) -> str:
    """Name the builder rule responsible for an automatic no-data insertion."""
    # ``ScndryOblgrIncm`` used to be named here. Since Phase 2 the builder does
    # not fill it: RREL20/RREL21 are declared in
    # the effective Annex 2 contract, so the path is sourced from
    # the prepared data and never reaches this attribution — and a run that
    # somehow lacked them would be refused by the builder, not filled.
    joined = "/".join(path)
    if "HstrclColltn" in joined:
        return ("the schema requires a 36-month historical collection series under "
                "non-performing data and the prepared data carried none "
                "(builder rule: _ensure_hstrcl_colltn_nd_defaults)")
    if "NonPrfrmgData" in joined:
        return ("the schema requires non-performing data on this branch and the "
                "prepared data carried none "
                "(builder rule: _ensure_nprf_nonprfrmgdata_defaults)")
    return ("the schema requires this element and the prepared data carried no "
            "value for it, so the builder inserted a no-data sentinel")


def _artefact_stem(request: DeliveryRequest) -> str:
    parts = [ANNEX_ID.lower(),
             str(request.portfolio_id or "portfolio"),
             str(request.reporting_date or "")]
    stem = "_".join(p for p in parts if p)
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem) or "annex2"


def _scan_csv(path: Path) -> Tuple[Optional[List[str]], int]:
    """Header and row count without materialising the frame.

    Preflight must be cheap: this is the check that decides whether to spend
    three minutes building a 200 MB file.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return None, 0
        if not header:
            return None, 0
        count = sum(1 for _ in reader)
    return [str(c).strip() for c in header], count


def _load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    import yaml

    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return None
    except yaml.YAMLError:  # pragma: no cover - malformed config
        return None


__all__ = ["Annex2Provider", "Annex2Paths", "ANNEX2_DEFINITION", "ANNEX_ID", "ANNEX_NAME"]
