from pathlib import Path

from lxml import etree
import pandas as pd

from engine.gate_5_delivery.xml_builder_annex2 import (
    MappingSpec,
    _alias_leaf_tag_for_xsd,
    _coerce_record_value_for_branch,
    _ensure_nprf_nonprfrmgdata_defaults,
    _ensure_scndry_oblgr_incm_defaults,
    _normalize_path_for_xsd,
    apply_record_code,
    build_annex2_tree,
    build_order_index,
    create_new_record_node,
    get_namespace_from_xsd,
    load_code_order,
    load_mapping_specs,
)


def test_branch_coercion_for_rrel12_non_year_value() -> None:
    assert _coerce_record_value_for_branch("RREL12", "GBZZZ") == "2026"
    assert _coerce_record_value_for_branch("RREL12", "2026") == "2026"
    assert _coerce_record_value_for_branch("RREL11", "GBZZZ") == "GBZZZ"


def test_alias_leaf_tag_for_xsd() -> None:
    assert _alias_leaf_tag_for_xsd("RREL83", "LEI") == "LEI"
    assert _alias_leaf_tag_for_xsd("RREC8", "Lien") == "Lien"
    assert _alias_leaf_tag_for_xsd("RREL1", "ScrtstnIdr") == "ScrtstnIdr"


def test_rrec8_lienval_renders_with_lien_child() -> None:
    ns = "urn:test"
    spec = MappingSpec(
        code="RREC8",
        tag="Lien",
        path="/Document/Root/UndrlygXpsrRcrd/Coll/CmonData/LienVal/Lien",
        multiplicity="[1..1]",
        pnp="PRF",
        row_idx=1,
    )
    specs_by_code = {"RREC8": [spec]}
    order_index = build_order_index(specs_by_code)
    root = etree.Element(f"{{{ns}}}Document")
    record = create_new_record_node(root, "/Document/Root/UndrlygXpsrRcrd", ns, order_index)

    row = pd.Series({"RREC8": "1"})
    apply_record_code(
        record,
        row,
        "RREC8",
        specs_by_code,
        "/Document/Root/UndrlygXpsrRcrd",
        ns,
        order_index,
        "GBP",
    )

    lien_vals = root.xpath("//*[local-name()='LienVal']")
    assert len(lien_vals) == 1
    lien_children = root.xpath("//*[local-name()='Lien']")
    assert len(lien_children) == 1
    assert lien_children[0].text == "1"


def test_normalize_path_for_lei_nodata_branch() -> None:
    raw = "/Document/Root/UndrlygXpsrRcrd/Orgntr/LEI/NoDataOptn/NoData"
    normalized = _normalize_path_for_xsd("RREL80", raw)
    assert normalized == "/Document/Root/UndrlygXpsrRcrd/Orgntr/LEICd/NoDataOptn/NoData"


def test_rrel20_secondary_income_builds_expected_container_chain() -> None:
    ns = "urn:test"
    spec = MappingSpec(
        code="RREL20",
        tag="Amt",
        path="/Document/Root/UndrlygXpsrRcrd/FinDtls/ScndryOblgrIncm/IncmVal/Val/Amt",
        multiplicity="[1..1]",
        pnp="PRF",
        row_idx=1,
    )
    specs_by_code = {"RREL20": [spec]}
    order_index = build_order_index(specs_by_code)
    root = etree.Element(f"{{{ns}}}Document")
    record = create_new_record_node(root, "/Document/Root/UndrlygXpsrRcrd", ns, order_index)

    row = pd.Series({"RREL20": "1234.56"})
    apply_record_code(
        record,
        row,
        "RREL20",
        specs_by_code,
        "/Document/Root/UndrlygXpsrRcrd",
        ns,
        order_index,
        "GBP",
    )

    assert len(root.xpath("//*[local-name()='ScndryOblgrIncm']")) == 1
    assert len(root.xpath("//*[local-name()='IncmVal']")) == 1
    amt_nodes = root.xpath("//*[local-name()='Amt']")
    assert len(amt_nodes) == 1
    assert amt_nodes[0].text == "1234.56"
    assert amt_nodes[0].get("Ccy") == "GBP"


def test_load_mapping_specs_includes_nprf_npe_codes(monkeypatch) -> None:
    df = pd.DataFrame(
        [
            {
                "RTS Field code": "NPEL7",
                "XML TAG": "<InRcvrshp>",
                "PATH": "/Document/Root/UndrlygXpsrRcrd/NonPrfrmgLn/UndrlygXpsrData/NonPrfrmgData/UndrlygXpsr/BrrwrDtls/InRcvrshp",
                "MULTIPLICITY": "[1..1]",
                "Template": "RRE",
                "Performing/Non Performing": "NPRF",
            }
        ]
    )
    monkeypatch.setattr("engine.gate_5_delivery.xml_builder_annex2.pd.read_excel", lambda *a, **k: df)

    specs = load_mapping_specs("dummy.xlsx", "sheet", "NPRF")
    assert "NPEL7" in specs


def test_ensure_scndry_oblgr_incm_defaults_adds_nd5_choice_nodes() -> None:
    ns = "urn:test"
    specs_by_code = {
        "RREL18": [
            MappingSpec(
                code="RREL18",
                tag="Amt",
                path="/Document/Root/UndrlygXpsrRcrd/UndrlygXpsrCmonData/FinDtls/MthlyIncm/Val/Amt",
                multiplicity="[1..1]",
                pnp="PRF",
                row_idx=1,
            )
        ],
        "RREL20": [
            MappingSpec(
                code="RREL20",
                tag="NoData",
                path="/Document/Root/UndrlygXpsrRcrd/UndrlygXpsrCmonData/FinDtls/ScndryOblgrIncm/IncmVal/NoDataOptn/NoData",
                multiplicity="[1..1]",
                pnp="PRF",
                row_idx=2,
            )
        ],
        "RREL21": [
            MappingSpec(
                code="RREL21",
                tag="NoData",
                path="/Document/Root/UndrlygXpsrRcrd/UndrlygXpsrCmonData/FinDtls/ScndryOblgrIncm/Vrfctn/NoDataOptn/NoData",
                multiplicity="[1..1]",
                pnp="PRF",
                row_idx=3,
            )
        ],
    }
    order_index = build_order_index(specs_by_code)
    root = etree.Element(f"{{{ns}}}Document")
    record = create_new_record_node(root, "/Document/Root/UndrlygXpsrRcrd", ns, order_index)

    fin = etree.SubElement(etree.SubElement(record, f"{{{ns}}}UndrlygXpsrCmonData"), f"{{{ns}}}FinDtls")
    mthly = etree.SubElement(fin, f"{{{ns}}}MthlyIncm")
    val = etree.SubElement(mthly, f"{{{ns}}}Val")
    etree.SubElement(val, f"{{{ns}}}Amt").text = "1"

    _ensure_scndry_oblgr_incm_defaults(record, ns, order_index)

    assert len(root.xpath("//*[local-name()='ScndryOblgrIncm']")) == 1
    assert len(root.xpath("//*[local-name()='IncmVal']/*[local-name()='NoDataOptn']/*[local-name()='NoData' and text()='ND5']")) == 1
    assert len(root.xpath("//*[local-name()='Vrfctn']/*[local-name()='NoDataOptn']/*[local-name()='NoData' and text()='ND5']")) == 1


def test_ensure_nprf_nonprfrmgdata_defaults_adds_required_containers() -> None:
    ns = "urn:test"
    specs_by_code = {
        "NPEL7": [
            MappingSpec(
                code="NPEL7",
                tag="NoData",
                path="/Document/Root/UndrlygXpsrRcrd/NonPrfrmgLn/UndrlygXpsrData/NonPrfrmgData/UndrlygXpsr/BrrwrDtls/InRcvrshp/NoDataOptn/NoData",
                multiplicity="[1..1]",
                pnp="NPRF",
                row_idx=1,
            )
        ],
        "NPEC14": [
            MappingSpec(
                code="NPEC14",
                tag="NoData",
                path="/Document/Root/UndrlygXpsrRcrd/NonPrfrmgLn/Coll/NonPrfrmgData/Val/OnMktPric/NoDataOptn/NoData",
                multiplicity="[1..1]",
                pnp="NPRF",
                row_idx=2,
            )
        ],
    }
    order_index = build_order_index(specs_by_code)
    root = etree.Element(f"{{{ns}}}Document")
    record = create_new_record_node(root, "/Document/Root/UndrlygXpsrRcrd", ns, order_index)
    non_prfrmg_ln = etree.SubElement(record, f"{{{ns}}}NonPrfrmgLn")
    etree.SubElement(non_prfrmg_ln, f"{{{ns}}}UndrlygXpsrData")
    etree.SubElement(non_prfrmg_ln, f"{{{ns}}}Coll")

    _ensure_nprf_nonprfrmgdata_defaults(
        record,
        specs_by_code,
        "/Document/Root/UndrlygXpsrRcrd",
        ns,
        order_index,
        "GBP",
        ["NPEL7", "NPEC14"],
    )

    assert len(root.xpath("//*[local-name()='UndrlygXpsrData']/*[local-name()='NonPrfrmgData']")) == 1
    assert len(root.xpath("//*[local-name()='Coll']/*[local-name()='NonPrfrmgData']")) == 1
    assert len(root.xpath("//*[local-name()='HstrclColltn']")) == 1


def test_ensure_nprf_nonprfrmgdata_defaults_not_triggered_for_prf_shapes() -> None:
    ns = "urn:test"
    specs_by_code = {
        "RREL20": [
            MappingSpec(
                code="RREL20",
                tag="Amt",
                path="/Document/Root/UndrlygXpsrRcrd/FinDtls/ScndryOblgrIncm/IncmVal/Val/Amt",
                multiplicity="[1..1]",
                pnp="PRF",
                row_idx=1,
            )
        ]
    }
    order_index = build_order_index(specs_by_code)
    root = etree.Element(f"{{{ns}}}Document")
    record = create_new_record_node(root, "/Document/Root/UndrlygXpsrRcrd", ns, order_index)
    prfrmg_ln = etree.SubElement(record, f"{{{ns}}}PrfrmgLn")
    etree.SubElement(prfrmg_ln, f"{{{ns}}}UndrlygXpsrData")
    etree.SubElement(prfrmg_ln, f"{{{ns}}}Coll")

    _ensure_nprf_nonprfrmgdata_defaults(
        record,
        specs_by_code,
        "/Document/Root/UndrlygXpsrRcrd",
        ns,
        order_index,
        "GBP",
        [],
    )

    assert len(root.xpath("//*[local-name()='NonPrfrmgData']")) == 0


def test_delivery_ready_fixture_without_npe_columns_still_xsd_valid_for_prf_and_nprf() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_csv = repo_root / "tests" / "fixtures" / "annex2_delivery_ready_no_npe.csv"
    workbook = repo_root / "DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report_Version_1.3.1.xlsx"
    code_order_yaml = repo_root / "config" / "system" / "esma_code_order.yaml"
    xsd = repo_root / "config" / "system" / "DRAFT1auth.099.001.04_1.3.0.xsd"

    df = pd.read_csv(fixture_csv, dtype=str).fillna("")
    assert not any(c.startswith("NPEL") for c in df.columns)
    assert not any(c.startswith("NPEC") for c in df.columns)

    code_order = load_code_order(str(code_order_yaml))
    ns = get_namespace_from_xsd(str(xsd))
    schema = etree.XMLSchema(etree.parse(str(xsd)))

    # PRF remains unaffected by NPRF-only fallbacks
    prf_specs = load_mapping_specs(str(workbook), "DRAFT1auth.099.001.04", "PRF")
    prf_root = build_annex2_tree(df, code_order, prf_specs, ns, "GBP", str(xsd))
    assert schema.validate(prf_root)
    assert len(prf_root.xpath("//*[local-name()='NonPrfrmgLn']")) == 0

    # NPRF uses fallback scaffolding and stays schema-valid despite no NPE columns.
    nprf_specs = load_mapping_specs(str(workbook), "DRAFT1auth.099.001.04", "NPRF")
    nprf_root = build_annex2_tree(df, code_order, nprf_specs, ns, "GBP", str(xsd))
    assert schema.validate(nprf_root)
    assert len(nprf_root.xpath("//*[local-name()='NonPrfrmgLn']/*[local-name()='UndrlygXpsrData']/*[local-name()='NonPrfrmgData']")) >= 1
    assert len(nprf_root.xpath("//*[local-name()='NonPrfrmgLn']/*[local-name()='Coll']/*[local-name()='NonPrfrmgData']")) >= 1
    assert len(nprf_root.xpath("//*[local-name()='HstrclColltn']")) >= 1
    assert len(
        nprf_root.xpath(
            "//*[local-name()='HstrclColltn']//*[starts-with(local-name(),'Mnth')]/*[local-name()='NoDataOptn']/*[local-name()='NoData' and text()='ND5']"
        )
    ) >= 1
