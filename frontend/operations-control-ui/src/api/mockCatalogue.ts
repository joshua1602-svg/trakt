/**
 * GENERATED from config/onboarding/field_catalogue.yaml — do not edit by hand.
 *
 * Regenerate with:
 *     python -m scripts.build_mock_catalogue
 *
 * The real browser reads this model from the API. This copy exists only so the
 * mock client can render the same wizard without a backend, and is generated
 * so the fixture cannot drift from the governed catalogue.
 */

import type { CaseKind, CaseStatus, Catalogue } from "./onboardingTypes";

export const CATALOGUE = {
  "version": 1,
  "sections": [
    {
      "key": "client",
      "label": "About the client",
      "help": "The organisation Trakt reports for. Its identity, where it operates and who Trakt should speak to.",
      "repeatable": false,
      "repeatable_key": "",
      "item_label_field": "",
      "min_items": 0,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "client_name",
          "label": "Client name",
          "scope": "client",
          "source": "client_supplied",
          "type": "text",
          "help": "How this client should be referred to in reports and in Trakt.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 120,
          "unique_across": "",
          "consumers": [
            "Client configuration",
            "every report heading"
          ],
          "writes_to": "client_config:client.display_name",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "client_id",
          "label": "Client identifier",
          "scope": "client",
          "source": "operator_supplied",
          "type": "identifier",
          "help": "Short, stable and unique. Used in storage locations and in every record Trakt keeps about this client. It cannot be changed later.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "identifier",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "clients",
          "consumers": [
            "Client configuration",
            "source registrations",
            "storage layout"
          ],
          "writes_to": "client_config:client.client_id",
          "evidence_required": false,
          "sensitive": false,
          "amendable": false,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "jurisdiction",
          "label": "Jurisdiction",
          "scope": "client",
          "source": "client_supplied",
          "type": "country",
          "help": "The country whose rules the client reports under.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "country",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration",
            "geographic enrichment"
          ],
          "writes_to": "client_config:portfolio.country",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "reporting_currency",
          "label": "Reporting currency",
          "scope": "client",
          "source": "trakt_default",
          "type": "currency",
          "help": "The currency amounts are reported in.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "currency",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration",
            "all monetary normalisation"
          ],
          "writes_to": "client_config:portfolio.base_currency",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "time_zone",
          "label": "Reporting time zone",
          "scope": "client",
          "source": "trakt_default",
          "type": "text",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Reporting calendar"
          ],
          "writes_to": "client_config:client.time_zone",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "environment",
          "label": "Environment",
          "scope": "operational",
          "source": "trakt_default",
          "type": "enum",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "production",
              "label": "Production"
            },
            {
              "value": "uat",
              "label": "Test"
            }
          ],
          "options_from": "",
          "default": "production",
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration"
          ],
          "writes_to": "client_config:client.environment",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "entities",
      "label": "Legal and reporting entities",
      "help": "The entities involved in the transaction. Capture each entity once and give it every role it holds — the same company is often both originator and servicer.",
      "repeatable": true,
      "repeatable_key": "entity_id",
      "item_label_field": "legal_name",
      "min_items": 1,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "entity_id",
          "label": "Entity reference",
          "scope": "legal_entity",
          "source": "system_generated",
          "type": "identifier",
          "help": "Generated by Trakt so portfolios can point at this entity.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": false,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "legal_name",
          "label": "Legal name",
          "scope": "legal_entity",
          "source": "client_supplied",
          "type": "text",
          "help": "The full registered name. Where the entity has an identifier, this must match the name held against it in the global register.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 200,
          "unique_across": "",
          "consumers": [
            "Annex 2 RREL82 where the entity is the originator",
            "Annex 12 IVSS3/IVSS4 where it is the reporting entity"
          ],
          "writes_to": "derived:client_config.defaults",
          "evidence_required": true,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "roles",
          "label": "Roles",
          "scope": "legal_entity",
          "source": "client_supplied",
          "type": "multi_enum",
          "help": "Every role this entity holds in the transaction.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "originator",
              "label": "Originator"
            },
            {
              "value": "sponsor",
              "label": "Sponsor"
            },
            {
              "value": "sspe",
              "label": "Issuer / SSPE"
            },
            {
              "value": "reporting_entity",
              "label": "Reporting entity"
            },
            {
              "value": "original_lender",
              "label": "Original lender"
            },
            {
              "value": "servicer",
              "label": "Servicer"
            },
            {
              "value": "risk_retention_holder",
              "label": "Risk retention holder"
            },
            {
              "value": "trustee",
              "label": "Trustee"
            },
            {
              "value": "warehouse_lender",
              "label": "Warehouse lender"
            },
            {
              "value": "investment_manager",
              "label": "Investment manager"
            },
            {
              "value": "calculation_agent",
              "label": "Calculation agent"
            },
            {
              "value": "seller",
              "label": "Seller"
            }
          ],
          "options_from": "entity_role",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Regime reporting",
            "report cover pages"
          ],
          "writes_to": "derived:client_config.reporting_parties",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "lei",
          "label": "Legal Entity Identifier",
          "scope": "legal_entity",
          "source": "client_supplied",
          "type": "lei",
          "help": "Twenty characters. Required for any entity named in regulatory reporting.",
          "required": false,
          "required_when": "roles contains originator",
          "asked_when": "",
          "validation": "lei",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Annex 2 RREL83",
            "Annex 12 IVSS1"
          ],
          "writes_to": "derived:client_config.defaults",
          "evidence_required": true,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "registration_number",
          "label": "Company registration number",
          "scope": "legal_entity",
          "source": "client_supplied",
          "type": "text",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": true,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "country_of_establishment",
          "label": "Country of establishment",
          "scope": "legal_entity",
          "source": "client_supplied",
          "type": "country",
          "help": "",
          "required": false,
          "required_when": "roles contains originator",
          "asked_when": "",
          "validation": "country",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Annex 2 RREL84"
          ],
          "writes_to": "derived:client_config.defaults",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "registered_address",
          "label": "Registered address",
          "scope": "legal_entity",
          "source": "client_supplied",
          "type": "multiline",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Report cover pages"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "contacts",
      "label": "Contacts and distribution",
      "help": "Who Trakt should contact, and who receives what.",
      "repeatable": false,
      "repeatable_key": "",
      "item_label_field": "",
      "min_items": 0,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "reporting_contact_name",
          "label": "Primary reporting contact",
          "scope": "contact",
          "source": "client_supplied",
          "type": "text",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration",
            "publication notification"
          ],
          "writes_to": "client_config:client.reporting_contact.name",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "reporting_contact_email",
          "label": "Reporting email",
          "scope": "contact",
          "source": "client_supplied",
          "type": "email",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "email",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration",
            "publication notification"
          ],
          "writes_to": "client_config:client.reporting_contact.email",
          "evidence_required": false,
          "sensitive": true,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "reporting_contact_phone",
          "label": "Reporting contact telephone",
          "scope": "contact",
          "source": "client_supplied",
          "type": "text",
          "help": "",
          "required": false,
          "required_when": "reporting.products contains investor_reporting",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Annex 12 IVSS6"
          ],
          "writes_to": "annex12_config:annex12.deal.IVSS6",
          "evidence_required": false,
          "sensitive": true,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "operational_contact_name",
          "label": "Operational contact",
          "scope": "contact",
          "source": "client_supplied",
          "type": "text",
          "help": "Who Trakt speaks to about deliveries and data questions.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration"
          ],
          "writes_to": "client_config:client.operational_contact.name",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "operational_contact_email",
          "label": "Operational email",
          "scope": "contact",
          "source": "client_supplied",
          "type": "email",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "email",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration"
          ],
          "writes_to": "client_config:client.operational_contact.email",
          "evidence_required": false,
          "sensitive": true,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "authorised_approver",
          "label": "Authorised approver",
          "scope": "contact",
          "source": "client_supplied",
          "type": "text",
          "help": "Who at the client may approve a report for release.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "investor_report_recipients",
          "label": "Investor report recipients",
          "scope": "contact",
          "source": "client_supplied",
          "type": "multiline",
          "help": "One address per line.",
          "required": false,
          "required_when": "reporting.products contains investor_reporting",
          "asked_when": "reporting.products contains investor_reporting",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Investor report distribution"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": true,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "portfolios",
      "label": "Portfolios",
      "help": "The books Trakt will report on. A portfolio exists from the moment it is approved here — it does not wait for a first delivery.",
      "repeatable": true,
      "repeatable_key": "portfolio_id",
      "item_label_field": "display_name",
      "min_items": 1,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "portfolio_id",
          "label": "Portfolio identifier",
          "scope": "portfolio",
          "source": "operator_supplied",
          "type": "identifier",
          "help": "Short, stable and unique within this client.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "identifier",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "portfolios",
          "consumers": [
            "Source registrations",
            "storage layout",
            "portfolio metadata"
          ],
          "writes_to": "portfolio_registry:source_portfolio_id",
          "evidence_required": false,
          "sensitive": false,
          "amendable": false,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "display_name",
          "label": "Portfolio name",
          "scope": "portfolio",
          "source": "client_supplied",
          "type": "text",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Portfolio metadata",
            "report headings"
          ],
          "writes_to": "portfolio_registry:source_portfolio_label",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "asset_class",
          "label": "Asset class",
          "scope": "portfolio",
          "source": "inferred",
          "type": "enum",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "equity_release",
              "label": "Equity Release"
            }
          ],
          "options_from": "asset_class",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration",
            "regime eligibility",
            "product profile"
          ],
          "writes_to": "client_config:portfolio.asset_class",
          "evidence_required": false,
          "sensitive": false,
          "amendable": false,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "portfolio_type",
          "label": "How the book was acquired",
          "scope": "portfolio",
          "source": "client_supplied",
          "type": "enum",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "direct",
              "label": "Originated by the client"
            },
            {
              "value": "acquired",
              "label": "Acquired from a third party"
            }
          ],
          "options_from": "portfolio_type",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Source registrations",
            "storage layout",
            "origination default"
          ],
          "writes_to": "portfolio_registry:source_portfolio_type",
          "evidence_required": false,
          "sensitive": false,
          "amendable": false,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "structure",
          "label": "How the book is held",
          "scope": "portfolio",
          "source": "operator_supplied",
          "type": "enum",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "on_balance_sheet",
              "label": "On balance sheet"
            },
            {
              "value": "warehouse",
              "label": "Warehouse facility"
            },
            {
              "value": "spv",
              "label": "Special purpose vehicle"
            },
            {
              "value": "managed",
              "label": "Managed for a third party"
            }
          ],
          "options_from": "portfolio_structure",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Portfolio metadata"
          ],
          "writes_to": "portfolio_registry:structure",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "owning_entity",
          "label": "Owning legal entity",
          "scope": "portfolio",
          "source": "inferred",
          "type": "entity_reference",
          "help": "Which of the entities above holds this book.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Portfolio metadata",
            "regime reporting"
          ],
          "writes_to": "portfolio_registry:owning_entity",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "reporting_currency",
          "label": "Portfolio reporting currency",
          "scope": "portfolio",
          "source": "inferred",
          "type": "currency",
          "help": "Leave blank to use the client's reporting currency.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "currency",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Portfolio metadata"
          ],
          "writes_to": "portfolio_registry:reporting_currency",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "originates",
          "label": "Still originating",
          "scope": "portfolio",
          "source": "derived",
          "type": "boolean",
          "help": "Whether new loans are still being written to this book.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "portfolio_type == direct",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Portfolio metadata",
            "forecast treatment"
          ],
          "writes_to": "portfolio_registry:originates",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "period_convention",
          "label": "Reporting period convention",
          "scope": "portfolio",
          "source": "inferred",
          "type": "enum",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "calendar_month_end",
              "label": "Calendar month end"
            },
            {
              "value": "calendar_quarter_end",
              "label": "Calendar quarter end"
            },
            {
              "value": "business_month_end",
              "label": "Last business day of the month"
            },
            {
              "value": "anniversary",
              "label": "Anniversary of the facility date"
            }
          ],
          "options_from": "period_convention",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Reporting calendar",
            "period validation"
          ],
          "writes_to": "portfolio_registry:period_convention",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "sources",
      "label": "Expected deliveries",
      "help": "What Trakt should expect to receive for each book, and how. One registration is created per book and dataset.",
      "repeatable": true,
      "repeatable_key": "source_key",
      "item_label_field": "source_key",
      "min_items": 0,
      "derived_from": "portfolios",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "portfolio_id",
          "label": "Portfolio",
          "scope": "source",
          "source": "derived",
          "type": "portfolio_reference",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "source_registry:source_portfolio_id",
          "evidence_required": false,
          "sensitive": false,
          "amendable": false,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "dataset",
          "label": "Book",
          "scope": "source",
          "source": "operator_supplied",
          "type": "enum",
          "help": "The funded book is the loans already advanced. Pipeline is what is being written.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "funded",
              "label": "Funded"
            },
            {
              "value": "pipeline",
              "label": "Pipeline"
            }
          ],
          "options_from": "dataset",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Source registration",
            "intake routing"
          ],
          "writes_to": "source_registry:dataset",
          "evidence_required": false,
          "sensitive": false,
          "amendable": false,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "cadence",
          "label": "Expected cadence",
          "scope": "source",
          "source": "trakt_default",
          "type": "enum",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "monthly",
              "label": "Monthly"
            },
            {
              "value": "weekly",
              "label": "Weekly"
            },
            {
              "value": "daily",
              "label": "Daily"
            },
            {
              "value": "adhoc",
              "label": "Adhoc"
            },
            {
              "value": "ad_hoc",
              "label": "Ad hoc"
            }
          ],
          "options_from": "cadence",
          "default": "monthly",
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Source registration",
            "storage layout"
          ],
          "writes_to": "source_registry:frequency",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "source_party",
          "label": "Who sends the data",
          "scope": "source",
          "source": "client_supplied",
          "type": "text",
          "help": "The system or organisation the files come from.",
          "required": false,
          "required_when": "",
          "asked_when": "portfolio_type == acquired",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Source registration"
          ],
          "writes_to": "source_registry:source_system",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "delivery_channel",
          "label": "How files arrive",
          "scope": "source",
          "source": "operator_supplied",
          "type": "enum",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "sftp",
              "label": "SFTP"
            },
            {
              "value": "blob_upload",
              "label": "Secure upload"
            },
            {
              "value": "email",
              "label": "Email"
            },
            {
              "value": "api",
              "label": "System to system"
            }
          ],
          "options_from": "delivery_channel",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "intake setup"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "file_format",
          "label": "File format",
          "scope": "source",
          "source": "inferred",
          "type": "enum",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "csv",
              "label": "CSV"
            },
            {
              "value": "xlsx",
              "label": "Excel"
            },
            {
              "value": "parquet",
              "label": "Parquet"
            },
            {
              "value": "fixed_width",
              "label": "Fixed width"
            }
          ],
          "options_from": "file_format",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "mapping"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "expected_files",
          "label": "Expected files",
          "scope": "source",
          "source": "inferred",
          "type": "multiline",
          "help": "The file names or patterns expected in each delivery, one per line. Used to tell an incomplete pack from a complete one.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Source registration",
            "pack completeness"
          ],
          "writes_to": "source_registry:expected_columns",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "cut_off_convention",
          "label": "Reporting cut-off",
          "scope": "source",
          "source": "derived",
          "type": "enum",
          "help": "The point in the period the data is taken as at.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "calendar_month_end",
              "label": "Calendar month end"
            },
            {
              "value": "calendar_quarter_end",
              "label": "Calendar quarter end"
            },
            {
              "value": "business_month_end",
              "label": "Last business day of the month"
            },
            {
              "value": "anniversary",
              "label": "Anniversary of the facility date"
            }
          ],
          "options_from": "period_convention",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "sample_provided",
          "label": "Sample file provided",
          "scope": "source",
          "source": "operator_supplied",
          "type": "boolean",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Readiness"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "mapping_complete",
          "label": "Mapping approved",
          "scope": "source",
          "source": "operator_supplied",
          "type": "boolean",
          "help": "Left unticked, the first delivery routes through source onboarding for mapping approval, which is the normal path for a new source.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Source registration status"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "regime_required",
          "label": "Included in regulatory reporting",
          "scope": "source",
          "source": "derived",
          "type": "boolean",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Source registration",
            "intake routing"
          ],
          "writes_to": "source_registry:regime_required",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "reporting",
      "label": "Reporting requirements",
      "help": "Which Trakt outputs this client receives. Where eligibility follows from the books, Trakt works it out; where it is a client decision, it is asked.",
      "repeatable": false,
      "repeatable_key": "",
      "item_label_field": "",
      "min_items": 0,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "products",
          "label": "Reporting products",
          "scope": "output",
          "source": "client_supplied",
          "type": "product_selection",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Client configuration",
            "regime configuration",
            "workflow outcomes"
          ],
          "writes_to": "derived:client_config.pipeline",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "regime",
      "label": "Regulatory information",
      "help": "Standing regulatory values. These stay the same from period to period, so Trakt asks once and reuses them on every delivery.",
      "repeatable": false,
      "repeatable_key": "",
      "item_label_field": "",
      "min_items": 0,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": true,
      "fields": [
        {
          "key": "originator_name",
          "label": "Originator name",
          "scope": "regime",
          "source": "derived",
          "type": "text",
          "help": "The full legal name of the originator. Must match the name held against the LEI in the GLEIF database.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "text",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 100,
          "unique_across": "",
          "consumers": [],
          "writes_to": "client_config:defaults.originator_name",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "RREL82",
          "product": "esma_annex2"
        },
        {
          "key": "originator_legal_entity_identifier",
          "label": "Originator LEI",
          "scope": "regime",
          "source": "derived",
          "type": "lei",
          "help": "The originator's 20-character Legal Entity Identifier.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "lei",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "client_config:defaults.originator_legal_entity_identifier",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "RREL83",
          "product": "esma_annex2"
        },
        {
          "key": "originator_establishment_country",
          "label": "Originator country of establishment",
          "scope": "regime",
          "source": "derived",
          "type": "country",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "country",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "client_config:defaults.originator_establishment_country",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "RREL84",
          "product": "esma_annex2"
        },
        {
          "key": "original_lender_legal_entity_identifier",
          "label": "Original lender LEI",
          "scope": "regime",
          "source": "derived",
          "type": "lei",
          "help": "Only where the original lender differs from the originator and is the same for the whole book. Left blank, the delivery rules apply ND5.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "lei",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "client_config:defaults.original_lender_legal_entity_identifier",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "RREL80",
          "product": "esma_annex2"
        },
        {
          "key": "original_lender_establishment_country",
          "label": "Original lender country of establishment",
          "scope": "regime",
          "source": "derived",
          "type": "country",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "country",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "client_config:defaults.original_lender_establishment_country",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "RREL81",
          "product": "esma_annex2"
        },
        {
          "key": "nuts_classification_year",
          "label": "NUTS classification year",
          "scope": "regime",
          "source": "trakt_default",
          "type": "enum",
          "help": "The NUTS vintage used for the regulatory geographic region fields.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "2021",
              "label": "2021"
            },
            {
              "value": "2024",
              "label": "2024"
            }
          ],
          "options_from": "",
          "default": "2021",
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "client_config:nuts_classification_year",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "esma_annex2"
        },
        {
          "key": "uk_geography_override",
          "label": "Report UK geography as GBZZZ",
          "scope": "regime",
          "source": "inferred",
          "type": "boolean",
          "help": "Delivers GBZZZ for UK obligor and collateral region (RREL11 / RREC6) while retaining granular ITL3 in canonical for MI. Applies to UK jurisdictions only.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "boolean",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "client_config:regime_overrides.ESMA_Annex2.uk_geography.enabled",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "esma_annex2"
        },
        {
          "key": "IVSS1",
          "label": "Securitisation unique identifier",
          "scope": "regime",
          "source": "derived",
          "type": "text",
          "help": "Left blank, the projector resolves this from the originator LEI in the client configuration.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "text",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS1",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS3",
          "label": "Securitisation name",
          "scope": "regime",
          "source": "derived",
          "type": "text",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "text",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS3",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS4",
          "label": "Reporting entity name",
          "scope": "regime",
          "source": "derived",
          "type": "text",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "text",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS4",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS5",
          "label": "Reporting entity contact person",
          "scope": "regime",
          "source": "derived",
          "type": "text",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "text",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS5",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS6",
          "label": "Reporting entity contact telephone",
          "scope": "regime",
          "source": "derived",
          "type": "text",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "text",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS6",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS7",
          "label": "Reporting entity contact email",
          "scope": "regime",
          "source": "derived",
          "type": "email",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "email",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS7",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS8",
          "label": "Risk retention method",
          "scope": "regime",
          "source": "client_supplied",
          "type": "enum",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "VSLC",
              "label": "Vertical slice"
            },
            {
              "value": "SLLS",
              "label": "Seller's share"
            },
            {
              "value": "RSEX",
              "label": "Randomly-selected exposures"
            },
            {
              "value": "FLTR",
              "label": "First loss tranche"
            },
            {
              "value": "FLEX",
              "label": "First loss exposure in each asset"
            },
            {
              "value": "NCOM",
              "label": "No compliance"
            },
            {
              "value": "OTHR",
              "label": "Other"
            }
          ],
          "options_from": "annex12_template:annex12.enums.IVSS8",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS8",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS9",
          "label": "Risk retention holder",
          "scope": "regime",
          "source": "derived",
          "type": "enum",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "ORIG",
              "label": "Originator"
            },
            {
              "value": "SPON",
              "label": "Sponsor"
            },
            {
              "value": "OLND",
              "label": "Original lender"
            },
            {
              "value": "SELL",
              "label": "Seller"
            },
            {
              "value": "NCOM",
              "label": "No compliance"
            },
            {
              "value": "OTHR",
              "label": "Other"
            }
          ],
          "options_from": "annex12_template:annex12.enums.IVSS9",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS9",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS10",
          "label": "Underlying exposure type",
          "scope": "regime",
          "source": "derived",
          "type": "enum",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "ALOL",
              "label": "Automobile loan or lease"
            },
            {
              "value": "CONL",
              "label": "Consumer loan"
            },
            {
              "value": "CMRT",
              "label": "Commercial mortgage"
            },
            {
              "value": "CCRR",
              "label": "Credit-card receivable"
            },
            {
              "value": "LEAS",
              "label": "Lease"
            },
            {
              "value": "RMRT",
              "label": "Residential mortgage"
            },
            {
              "value": "MIXD",
              "label": "Mixed"
            },
            {
              "value": "SMEL",
              "label": "SME"
            },
            {
              "value": "NSML",
              "label": "Non-SME corporate"
            },
            {
              "value": "OTHR",
              "label": "Other"
            }
          ],
          "options_from": "annex12_template:annex12.enums.IVSS10",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS10",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS11",
          "label": "Risk transfer method applied",
          "scope": "regime",
          "source": "client_supplied",
          "type": "yes_no",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "yes_no",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS11",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS12",
          "label": "Trigger measurements or ratios reported",
          "scope": "regime",
          "source": "client_supplied",
          "type": "yes_no",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "yes_no",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS12",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS13",
          "label": "Revolving / ramp-up end date",
          "scope": "regime",
          "source": "client_supplied",
          "type": "date_or_nd",
          "help": "A date where one applies, otherwise ND5.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "date_or_nd",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS13",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS20",
          "label": "Excess spread trapping mechanism",
          "scope": "regime",
          "source": "client_supplied",
          "type": "yes_no",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "yes_no",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS20",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        },
        {
          "key": "IVSS30",
          "label": "Risk weight approach",
          "scope": "regime",
          "source": "client_supplied",
          "type": "enum",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "STND",
              "label": "Standardised approach"
            },
            {
              "value": "FIRB",
              "label": "Foundation IRB"
            },
            {
              "value": "ADIR",
              "label": "Advanced IRB"
            }
          ],
          "options_from": "annex12_template:annex12.enums.IVSS30",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [],
          "writes_to": "annex12_config:annex12.deal.IVSS30",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": "investor_reporting"
        }
      ]
    },
    {
      "key": "presentation",
      "label": "Report presentation",
      "help": "How reports look, and the standing conventions behind the numbers. None of this is regulatory.",
      "repeatable": false,
      "repeatable_key": "",
      "item_label_field": "",
      "min_items": 0,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "report_title",
          "label": "Report title",
          "scope": "branding",
          "source": "inferred",
          "type": "text",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Management information dashboard"
          ],
          "writes_to": "client_config:mi.branding.app_title",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "brand_colour",
          "label": "Brand colour",
          "scope": "branding",
          "source": "client_supplied",
          "type": "colour",
          "help": "A hex colour, for example #1F3B5C.",
          "required": false,
          "required_when": "",
          "asked_when": "reporting.products contains mi",
          "validation": "colour",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Management information dashboard"
          ],
          "writes_to": "client_config:mi.branding.theme.primary_color",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "logo_uri",
          "label": "Logo",
          "scope": "branding",
          "source": "client_supplied",
          "type": "text",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "reporting.products contains mi",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Report cover pages"
          ],
          "writes_to": "client_config:mi.branding.logo_uri",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "disclaimer",
          "label": "Disclaimer",
          "scope": "branding",
          "source": "client_supplied",
          "type": "multiline",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "reporting.products contains mi",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Report cover pages"
          ],
          "writes_to": "client_config:mi.branding.disclaimer",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "day_count_convention",
          "label": "Day-count convention",
          "scope": "operational",
          "source": "inferred",
          "type": "enum",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "ACT365",
              "label": "Actual/365"
            },
            {
              "value": "ACT360",
              "label": "Actual/360"
            },
            {
              "value": "ACT_ACT",
              "label": "Actual/Actual"
            },
            {
              "value": "THIRTY_360",
              "label": "30/360"
            }
          ],
          "options_from": "day_count_convention",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Interest calculations"
          ],
          "writes_to": "client_config:loan_engine.day_count_convention",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "payment_frequency",
          "label": "Payment frequency",
          "scope": "operational",
          "source": "inferred",
          "type": "enum",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "MONTHLY",
              "label": "Monthly"
            },
            {
              "value": "QUARTERLY",
              "label": "Quarterly"
            },
            {
              "value": "SEMI_ANNUAL",
              "label": "Semi-annual"
            },
            {
              "value": "ANNUAL",
              "label": "Annual"
            },
            {
              "value": "AT_REDEMPTION",
              "label": "At redemption"
            }
          ],
          "options_from": "payment_frequency",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Interest calculations"
          ],
          "writes_to": "client_config:loan_engine.interest_payment_frequency",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "reporting_calendar_note",
          "label": "Reporting calendar",
          "scope": "operational",
          "source": "client_supplied",
          "type": "multiline",
          "help": "Anything Trakt should know about when reports are due — for example a fixed business day, or a quarter-end dependency.",
          "required": false,
          "required_when": "",
          "asked_when": "reporting.products contains mi",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "data_semantics",
      "label": "What your numbers mean",
      "help": "The business conventions behind the figures. These are properties of how the CLIENT runs their book, not of any particular file, so they are asked during onboarding and reused for every delivery afterwards.\nTrakt cannot work these out by reading data: two lenders can send the same column name and mean different things by it. What Trakt CAN work out — the format, the column names, the mapping — it does not ask for. Those live in \"How to read each file\", which is deferred until a file exists.",
      "repeatable": false,
      "repeatable_key": "",
      "item_label_field": "",
      "min_items": 0,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "balance_definition",
          "label": "What a balance means",
          "scope": "client",
          "source": "client_supplied",
          "type": "multiline",
          "help": "Whether balances include accrued interest, fees or arrears, and at what point in the month they are struck.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 2000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "first-delivery mapping review",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "gross_net_convention",
          "label": "Gross or net amounts",
          "scope": "client",
          "source": "client_supplied",
          "type": "multiline",
          "help": "Whether the amounts you report are gross or net, and of what — fees, charges, recoveries.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 2000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "units_and_currency",
          "label": "Units and currency",
          "scope": "client",
          "source": "client_supplied",
          "type": "text",
          "help": "Whether amounts are in units, thousands or millions, and in which currency.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 200,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "cut_off_convention",
          "label": "Reporting cut-off",
          "scope": "client",
          "source": "client_supplied",
          "type": "multiline",
          "help": "The point at which a reporting period closes for you — calendar month end, a fixed business day, or something else.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 2000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "measure_basis",
          "label": "Point in time or cumulative",
          "scope": "client",
          "source": "client_supplied",
          "type": "enum",
          "help": "Whether the figures you send describe a moment, a period, or a running total.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "point_in_time",
              "label": "Point in time, as at the reporting date"
            },
            {
              "value": "cumulative",
              "label": "Cumulative since inception"
            },
            {
              "value": "period",
              "label": "For the reporting period only"
            },
            {
              "value": "mixed",
              "label": "A mixture — described above"
            }
          ],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "cashflow_basis",
          "label": "Actual or forecast cashflows",
          "scope": "client",
          "source": "client_supplied",
          "type": "enum",
          "help": "Whether the cashflows you send are what actually happened, what you expect to happen, or both — and how the two are distinguished.",
          "required": false,
          "required_when": "",
          "asked_when": "reporting.products contains mi",
          "validation": "enum",
          "options": [
            {
              "value": "actual",
              "label": "Actual, as executed"
            },
            {
              "value": "forecast",
              "label": "Forecast or expected"
            },
            {
              "value": "both",
              "label": "Both, distinguished in the file"
            }
          ],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "valuation_basis",
          "label": "How valuations are struck",
          "scope": "client",
          "source": "client_supplied",
          "type": "multiline",
          "help": "Whose valuation it is, how old it may be, and whether it is indexed between full valuations.",
          "required": false,
          "required_when": "",
          "asked_when": "reporting.products contains mi",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 2000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "accrued_interest_treatment",
          "label": "How accrued interest is treated",
          "scope": "client",
          "source": "client_supplied",
          "type": "multiline",
          "help": "Whether interest is rolled into the balance, reported separately, or both — and from which date it accrues.",
          "required": false,
          "required_when": "",
          "asked_when": "reporting.products contains mi",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 2000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "redemption_definition",
          "label": "When a loan counts as redeemed",
          "scope": "client",
          "source": "client_supplied",
          "type": "multiline",
          "help": "What you treat as a redemption or completion, and the date you use for it.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 2000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "status_definitions",
          "label": "What your account statuses mean",
          "scope": "client",
          "source": "client_supplied",
          "type": "multiline",
          "help": "Any status of your own — arrears bands, holds, forbearance — and what each one means.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 4000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "data_definitions",
      "label": "How to read each file",
      "help": "File-specific technical detail, asked at the first delivery rather than during onboarding. Trakt learns the canonical mapping, the datatypes, the aliases and the schema fingerprint from the file itself, through the existing mapping path; what is asked here is only what remains once it has, and only about a file that exists.",
      "repeatable": true,
      "repeatable_key": "data_definitions",
      "item_label_field": "source_file",
      "min_items": 0,
      "derived_from": "",
      "deferred_until": "the client has sent a representative file, because these describe a FILE rather than the business behind it",
      "from_regime": false,
      "fields": [
        {
          "key": "source_file",
          "label": "Source file",
          "scope": "delivery",
          "source": "client_supplied",
          "type": "text",
          "help": "The file or extract this describes, as the client names it.",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 200,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "first-delivery mapping review"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "description",
          "label": "What the file contains",
          "scope": "delivery",
          "source": "client_supplied",
          "type": "multiline",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 2000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "first-delivery mapping review"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "proprietary_fields",
          "label": "Source columns that need explaining",
          "scope": "delivery",
          "source": "client_supplied",
          "type": "multiline",
          "help": "Any column in THIS file whose name would not tell Trakt what it holds, with the client's own definition. The business meaning behind the numbers is asked during onboarding, not here.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 4000,
          "unique_across": "",
          "consumers": [
            "First-delivery mapping review"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "date_conventions",
          "label": "Date format and datatype notes",
          "scope": "delivery",
          "source": "client_supplied",
          "type": "text",
          "help": "The date format used in this file, and anything about how values are typed that Trakt should not have to guess.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 200,
          "unique_across": "",
          "consumers": [
            "First-delivery mapping review"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "known_limitations",
          "label": "Known data-quality behaviour in this file",
          "scope": "delivery",
          "source": "client_supplied",
          "type": "multiline",
          "help": "Anything in this file the client already knows is missing, estimated or unreliable.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 4000,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "validation review"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    },
    {
      "key": "access",
      "label": "Who needs access",
      "help": "The people who will use Trakt for this client, and what each of them needs. Trakt records the requirement; provisioning is an operator action outside onboarding.",
      "repeatable": true,
      "repeatable_key": "access",
      "item_label_field": "user_name",
      "min_items": 0,
      "derived_from": "",
      "deferred_until": "",
      "from_regime": false,
      "fields": [
        {
          "key": "user_name",
          "label": "Name",
          "scope": "access",
          "source": "client_supplied",
          "type": "text",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 120,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "access provisioning"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "user_email",
          "label": "Email address",
          "scope": "access",
          "source": "client_supplied",
          "type": "email",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "email",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Onboarding record",
            "access provisioning"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "user_role",
          "label": "Role",
          "scope": "access",
          "source": "client_supplied",
          "type": "enum",
          "help": "",
          "required": true,
          "required_when": "",
          "asked_when": "",
          "validation": "enum",
          "options": [
            {
              "value": "viewer",
              "label": "Views reports and dashboards"
            },
            {
              "value": "analyst",
              "label": "Works with the data"
            },
            {
              "value": "approver",
              "label": "Approves deliveries and configuration"
            },
            {
              "value": "administrator",
              "label": "Administers this client's settings"
            }
          ],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Access provisioning"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "scope_note",
          "label": "Which clients or portfolios",
          "scope": "access",
          "source": "client_supplied",
          "type": "text",
          "help": "Leave blank for all of this client's portfolios.",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": 200,
          "unique_across": "",
          "consumers": [
            "Access provisioning"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "occ_access_required",
          "label": "Needs Operations Control Centre access",
          "scope": "access",
          "source": "client_supplied",
          "type": "boolean",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Access provisioning"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "dashboard_access_required",
          "label": "Needs dashboard access",
          "scope": "access",
          "source": "client_supplied",
          "type": "boolean",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Access provisioning"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        },
        {
          "key": "report_recipient",
          "label": "Receives reports by email",
          "scope": "access",
          "source": "client_supplied",
          "type": "boolean",
          "help": "",
          "required": false,
          "required_when": "",
          "asked_when": "",
          "validation": "",
          "options": [],
          "options_from": "",
          "default": null,
          "derived_default": "",
          "max_length": null,
          "unique_across": "",
          "consumers": [
            "Report distribution"
          ],
          "writes_to": "onboarding_record",
          "evidence_required": false,
          "sensitive": false,
          "amendable": true,
          "regime_code": "",
          "product": ""
        }
      ]
    }
  ],
  "vocabularies": {
    "asset_class": {
      "label": "Asset class",
      "help": "",
      "owner": "operations_control.configuration.packages.ASSET_MODEL",
      "values": [
        {
          "value": "equity_release",
          "label": "Equity Release"
        }
      ]
    },
    "dataset": {
      "label": "Book",
      "help": "",
      "owner": "operations_control.contracts.BATCH_DATASETS",
      "values": [
        {
          "value": "funded",
          "label": "Funded"
        },
        {
          "value": "pipeline",
          "label": "Pipeline"
        }
      ]
    },
    "cadence": {
      "label": "Reporting cadence",
      "help": "",
      "owner": "apps.blob_trigger_app.path_parser.VALID_FREQUENCIES",
      "values": [
        {
          "value": "monthly",
          "label": "Monthly"
        },
        {
          "value": "weekly",
          "label": "Weekly"
        },
        {
          "value": "daily",
          "label": "Daily"
        },
        {
          "value": "adhoc",
          "label": "Adhoc"
        },
        {
          "value": "ad_hoc",
          "label": "Ad hoc"
        }
      ]
    },
    "portfolio_type": {
      "label": "How the book was acquired",
      "help": "",
      "owner": "",
      "values": [
        {
          "value": "direct",
          "label": "Originated by the client"
        },
        {
          "value": "acquired",
          "label": "Acquired from a third party"
        }
      ]
    },
    "portfolio_structure": {
      "label": "How the book is held",
      "help": "",
      "owner": "",
      "values": [
        {
          "value": "on_balance_sheet",
          "label": "On balance sheet"
        },
        {
          "value": "warehouse",
          "label": "Warehouse facility"
        },
        {
          "value": "spv",
          "label": "Special purpose vehicle"
        },
        {
          "value": "managed",
          "label": "Managed for a third party"
        }
      ]
    },
    "entity_role": {
      "label": "Role",
      "help": "One legal entity may hold several roles. Roles are recorded against the entity, so its identity is captured once.",
      "owner": "",
      "values": [
        {
          "value": "originator",
          "label": "Originator"
        },
        {
          "value": "sponsor",
          "label": "Sponsor"
        },
        {
          "value": "sspe",
          "label": "Issuer / SSPE"
        },
        {
          "value": "reporting_entity",
          "label": "Reporting entity"
        },
        {
          "value": "original_lender",
          "label": "Original lender"
        },
        {
          "value": "servicer",
          "label": "Servicer"
        },
        {
          "value": "risk_retention_holder",
          "label": "Risk retention holder"
        },
        {
          "value": "trustee",
          "label": "Trustee"
        },
        {
          "value": "warehouse_lender",
          "label": "Warehouse lender"
        },
        {
          "value": "investment_manager",
          "label": "Investment manager"
        },
        {
          "value": "calculation_agent",
          "label": "Calculation agent"
        },
        {
          "value": "seller",
          "label": "Seller"
        }
      ]
    },
    "delivery_channel": {
      "label": "How files arrive",
      "help": "",
      "owner": "",
      "values": [
        {
          "value": "sftp",
          "label": "SFTP"
        },
        {
          "value": "blob_upload",
          "label": "Secure upload"
        },
        {
          "value": "email",
          "label": "Email"
        },
        {
          "value": "api",
          "label": "System to system"
        }
      ]
    },
    "file_format": {
      "label": "File format",
      "help": "",
      "owner": "",
      "values": [
        {
          "value": "csv",
          "label": "CSV"
        },
        {
          "value": "xlsx",
          "label": "Excel"
        },
        {
          "value": "parquet",
          "label": "Parquet"
        },
        {
          "value": "fixed_width",
          "label": "Fixed width"
        }
      ]
    },
    "period_convention": {
      "label": "Reporting period convention",
      "help": "",
      "owner": "",
      "values": [
        {
          "value": "calendar_month_end",
          "label": "Calendar month end"
        },
        {
          "value": "calendar_quarter_end",
          "label": "Calendar quarter end"
        },
        {
          "value": "business_month_end",
          "label": "Last business day of the month"
        },
        {
          "value": "anniversary",
          "label": "Anniversary of the facility date"
        }
      ]
    },
    "day_count_convention": {
      "label": "Day-count convention",
      "help": "",
      "owner": "",
      "values": [
        {
          "value": "ACT365",
          "label": "Actual/365"
        },
        {
          "value": "ACT360",
          "label": "Actual/360"
        },
        {
          "value": "ACT_ACT",
          "label": "Actual/Actual"
        },
        {
          "value": "THIRTY_360",
          "label": "30/360"
        }
      ]
    },
    "payment_frequency": {
      "label": "Payment frequency",
      "help": "",
      "owner": "",
      "values": [
        {
          "value": "MONTHLY",
          "label": "Monthly"
        },
        {
          "value": "QUARTERLY",
          "label": "Quarterly"
        },
        {
          "value": "SEMI_ANNUAL",
          "label": "Semi-annual"
        },
        {
          "value": "ANNUAL",
          "label": "Annual"
        },
        {
          "value": "AT_REDEMPTION",
          "label": "At redemption"
        }
      ]
    }
  },
  "not_collected": [
    {
      "key": "reporting_period",
      "label": "Reporting period",
      "source": "delivery_specific",
      "reason": "Supplied with each delivery. The workflow already asks for it, and a standing value would silently mis-date a report."
    },
    {
      "key": "static_reporting_date",
      "label": "Static reporting date",
      "source": "delivery_specific",
      "reason": "Present in the legacy client configuration but delivery-specific. Not written by onboarding."
    },
    {
      "key": "expected_schema_fingerprint",
      "label": "Expected file signature",
      "source": "inferred",
      "reason": "Learned from the first real delivery and pinned by the existing promotion path. Asking for it would be asking the operator to guess."
    },
    {
      "key": "file_role_schemas",
      "label": "File role signatures",
      "source": "inferred",
      "reason": "Learned at mapping approval from a representative pack."
    },
    {
      "key": "blob_prefix",
      "label": "Delivery location",
      "source": "system_generated",
      "reason": "Derived from the client, book type, dataset, cadence and portfolio using the production storage layout. Shown for confirmation, never typed."
    },
    {
      "key": "onboarding_case_id",
      "label": "Onboarding reference",
      "source": "system_generated",
      "reason": "Minted when the case is opened."
    },
    {
      "key": "nd_defaults",
      "label": "No-data policy",
      "source": "trakt_default",
      "reason": "The governed regime delivery rules carry no-data defaults. A client override is an administrator change, not an onboarding question."
    }
  ],
  "regime_products": {
    "mi": {
      "label": "Management Information",
      "always_applies": true,
      "derived_from": "Every governed delivery produces MI. The workflow outcome vocabulary (operations_control.contracts.OUTCOMES) has no MI-less option.",
      "fields": []
    },
    "esma_annex2": {
      "label": "ESMA Annex 2",
      "requires_dataset": "funded",
      "supported_by_assets_declaring": "ESMA_Annex2",
      "fields": [
        {
          "key": "originator_name",
          "label": "Originator name",
          "regime_code": "RREL82",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the entity holding the originator role",
          "required": false,
          "format": "text",
          "max_length": 100,
          "help": "The full legal name of the originator. Must match the name held against the LEI in the GLEIF database.",
          "writes_to": "client_config:defaults.originator_name"
        },
        {
          "key": "originator_legal_entity_identifier",
          "label": "Originator LEI",
          "regime_code": "RREL83",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the entity holding the originator role",
          "required": false,
          "format": "lei",
          "help": "The originator's 20-character Legal Entity Identifier.",
          "writes_to": "client_config:defaults.originator_legal_entity_identifier"
        },
        {
          "key": "originator_establishment_country",
          "label": "Originator country of establishment",
          "regime_code": "RREL84",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the entity holding the originator role",
          "required": false,
          "format": "country",
          "writes_to": "client_config:defaults.originator_establishment_country"
        },
        {
          "key": "original_lender_legal_entity_identifier",
          "label": "Original lender LEI",
          "regime_code": "RREL80",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the entity holding the original lender role",
          "required": false,
          "format": "lei",
          "help": "Only where the original lender differs from the originator and is the same for the whole book. Left blank, the delivery rules apply ND5.",
          "writes_to": "client_config:defaults.original_lender_legal_entity_identifier"
        },
        {
          "key": "original_lender_establishment_country",
          "label": "Original lender country of establishment",
          "regime_code": "RREL81",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the entity holding the original lender role",
          "required": false,
          "format": "country",
          "writes_to": "client_config:defaults.original_lender_establishment_country"
        },
        {
          "key": "nuts_classification_year",
          "label": "NUTS classification year",
          "classification": "standing_portfolio",
          "source": "trakt_default",
          "derives_from": "the governed default vintage",
          "required": false,
          "format": "enum",
          "default": "2021",
          "options": [
            "2021",
            "2024"
          ],
          "help": "The NUTS vintage used for the regulatory geographic region fields.",
          "writes_to": "client_config:nuts_classification_year"
        },
        {
          "key": "uk_geography_override",
          "label": "Report UK geography as GBZZZ",
          "classification": "standing_portfolio",
          "source": "inferred",
          "derives_from": "the client's jurisdiction",
          "required": false,
          "format": "boolean",
          "help": "Delivers GBZZZ for UK obligor and collateral region (RREL11 / RREC6) while retaining granular ITL3 in canonical for MI. Applies to UK jurisdictions only.",
          "writes_to": "client_config:regime_overrides.ESMA_Annex2.uk_geography.enabled"
        }
      ],
      "unrepresented": [
        {
          "key": "sponsor_name",
          "label": "Sponsor",
          "reason": "Annex 2 is the underlying-exposure report; it carries no sponsor field. Sponsor appears only as a risk-retention holder value in Annex 12 (IVSS9 = SPON)."
        },
        {
          "key": "sspe_name",
          "label": "SSPE",
          "reason": "No SSPE field exists in the Annex 2 field universe. The securitisation entity is named in Annex 12 (IVSS3 / IVSS4)."
        },
        {
          "key": "national_competent_authority",
          "label": "National competent authority",
          "reason": "Trakt produces the report; it does not file it. No NCA field exists in the field universe or the delivery rules."
        },
        {
          "key": "sts_status",
          "label": "STS status and notification identifier",
          "reason": "No STS fields exist in the current Annex 2 model. Adding them means extending the field universe, which is an administrator change."
        },
        {
          "key": "servicer_name",
          "label": "Servicer",
          "reason": "`source_system` on a source registration records who supplies the data. There is no regulatory servicer field in the Annex 2 model."
        },
        {
          "key": "calculation_conventions",
          "label": "Calculation conventions",
          "reason": "Day-count and payment-frequency conventions exist under `loan_engine:` in the client configuration and are captured in onboarding as static reporting information, but they are not Annex 2 fields."
        }
      ]
    },
    "investor_reporting": {
      "label": "Investor Reporting",
      "regime_id": "ESMA_Annex12",
      "fields": [
        {
          "key": "IVSS1",
          "label": "Securitisation unique identifier",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the reporting entity's identifier",
          "required": false,
          "format": "text",
          "help": "Left blank, the projector resolves this from the originator LEI in the client configuration.",
          "writes_to": "annex12_config:annex12.deal.IVSS1"
        },
        {
          "key": "IVSS3",
          "label": "Securitisation name",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the reporting entity's legal name",
          "required": false,
          "format": "text",
          "writes_to": "annex12_config:annex12.deal.IVSS3"
        },
        {
          "key": "IVSS4",
          "label": "Reporting entity name",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the reporting entity's legal name",
          "required": false,
          "format": "text",
          "writes_to": "annex12_config:annex12.deal.IVSS4"
        },
        {
          "key": "IVSS5",
          "label": "Reporting entity contact person",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the reporting contact",
          "required": false,
          "format": "text",
          "writes_to": "annex12_config:annex12.deal.IVSS5"
        },
        {
          "key": "IVSS6",
          "label": "Reporting entity contact telephone",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the reporting contact",
          "required": false,
          "format": "text",
          "writes_to": "annex12_config:annex12.deal.IVSS6"
        },
        {
          "key": "IVSS7",
          "label": "Reporting entity contact email",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the reporting contact",
          "required": false,
          "format": "email",
          "writes_to": "annex12_config:annex12.deal.IVSS7"
        },
        {
          "key": "IVSS8",
          "label": "Risk retention method",
          "classification": "standing_client",
          "required": true,
          "format": "enum",
          "options_from": "annex12_template:annex12.enums.IVSS8",
          "writes_to": "annex12_config:annex12.deal.IVSS8",
          "options": [
            {
              "value": "VSLC",
              "label": "Vertical slice"
            },
            {
              "value": "SLLS",
              "label": "Seller's share"
            },
            {
              "value": "RSEX",
              "label": "Randomly-selected exposures"
            },
            {
              "value": "FLTR",
              "label": "First loss tranche"
            },
            {
              "value": "FLEX",
              "label": "First loss exposure in each asset"
            },
            {
              "value": "NCOM",
              "label": "No compliance"
            },
            {
              "value": "OTHR",
              "label": "Other"
            }
          ]
        },
        {
          "key": "IVSS9",
          "label": "Risk retention holder",
          "classification": "standing_client",
          "source": "derived",
          "derives_from": "the role of the entity holding the retention",
          "required": false,
          "format": "enum",
          "options_from": "annex12_template:annex12.enums.IVSS9",
          "writes_to": "annex12_config:annex12.deal.IVSS9",
          "options": [
            {
              "value": "ORIG",
              "label": "Originator"
            },
            {
              "value": "SPON",
              "label": "Sponsor"
            },
            {
              "value": "OLND",
              "label": "Original lender"
            },
            {
              "value": "SELL",
              "label": "Seller"
            },
            {
              "value": "NCOM",
              "label": "No compliance"
            },
            {
              "value": "OTHR",
              "label": "Other"
            }
          ]
        },
        {
          "key": "IVSS10",
          "label": "Underlying exposure type",
          "classification": "standing_portfolio",
          "source": "derived",
          "derives_from": "the portfolio's asset class",
          "required": false,
          "format": "enum",
          "options_from": "annex12_template:annex12.enums.IVSS10",
          "writes_to": "annex12_config:annex12.deal.IVSS10",
          "options": [
            {
              "value": "ALOL",
              "label": "Automobile loan or lease"
            },
            {
              "value": "CONL",
              "label": "Consumer loan"
            },
            {
              "value": "CMRT",
              "label": "Commercial mortgage"
            },
            {
              "value": "CCRR",
              "label": "Credit-card receivable"
            },
            {
              "value": "LEAS",
              "label": "Lease"
            },
            {
              "value": "RMRT",
              "label": "Residential mortgage"
            },
            {
              "value": "MIXD",
              "label": "Mixed"
            },
            {
              "value": "SMEL",
              "label": "SME"
            },
            {
              "value": "NSML",
              "label": "Non-SME corporate"
            },
            {
              "value": "OTHR",
              "label": "Other"
            }
          ]
        },
        {
          "key": "IVSS11",
          "label": "Risk transfer method applied",
          "classification": "standing_client",
          "required": false,
          "format": "yes_no",
          "writes_to": "annex12_config:annex12.deal.IVSS11"
        },
        {
          "key": "IVSS12",
          "label": "Trigger measurements or ratios reported",
          "classification": "standing_client",
          "required": false,
          "format": "yes_no",
          "writes_to": "annex12_config:annex12.deal.IVSS12"
        },
        {
          "key": "IVSS13",
          "label": "Revolving / ramp-up end date",
          "classification": "standing_client",
          "required": false,
          "format": "date_or_nd",
          "help": "A date where one applies, otherwise ND5.",
          "writes_to": "annex12_config:annex12.deal.IVSS13"
        },
        {
          "key": "IVSS20",
          "label": "Excess spread trapping mechanism",
          "classification": "standing_client",
          "required": false,
          "format": "yes_no",
          "writes_to": "annex12_config:annex12.deal.IVSS20"
        },
        {
          "key": "IVSS30",
          "label": "Risk weight approach",
          "classification": "standing_client",
          "required": false,
          "format": "enum",
          "options_from": "annex12_template:annex12.enums.IVSS30",
          "writes_to": "annex12_config:annex12.deal.IVSS30",
          "options": [
            {
              "value": "STND",
              "label": "Standardised approach"
            },
            {
              "value": "FIRB",
              "label": "Foundation IRB"
            },
            {
              "value": "ADIR",
              "label": "Advanced IRB"
            }
          ]
        }
      ],
      "unrepresented": [
        {
          "key": "trustee_name",
          "label": "Trustee",
          "reason": "Captured as static reporting information in the onboarding profile (it appears on investor report covers), but Annex 12 has no trustee field, so it is not written into the regime configuration."
        },
        {
          "key": "IVSR_triggers",
          "label": "Trigger definitions (IVSR1-IVSR10)",
          "reason": "Currently hard-coded per deal in config/client/config_client_annex12.yaml as forced XML-structure values. They are deal-structure information, not client standing information, and are left where they are until a governed trigger model exists."
        },
        {
          "key": "IVSF_cashflows",
          "label": "Cashflow items (IVSF1-IVSF6)",
          "reason": "Period cashflow values. Delivery-specific by definition."
        }
      ]
    },
    "static_pools": {
      "label": "Static Pools",
      "requires_dataset": "funded",
      "fields": [],
      "unrepresented": [
        {
          "key": "cohort_definition",
          "label": "Cohort definition",
          "reason": "Held as asset configuration (static_pools_config_erm.yaml) and administered through the governed asset package, not per client."
        }
      ]
    }
  }
} as unknown as Catalogue;

export const STEPS: { key: string; label: string }[] = [
  {
    "key": "client",
    "label": "About the client"
  },
  {
    "key": "entities",
    "label": "Legal and reporting entities"
  },
  {
    "key": "contacts",
    "label": "Contacts and distribution"
  },
  {
    "key": "portfolios",
    "label": "Portfolios"
  },
  {
    "key": "sources",
    "label": "Expected deliveries"
  },
  {
    "key": "reporting",
    "label": "Reporting requirements"
  },
  {
    "key": "regime",
    "label": "Regulatory information"
  },
  {
    "key": "data_semantics",
    "label": "What your numbers mean"
  },
  {
    "key": "data_definitions",
    "label": "How to read each file"
  },
  {
    "key": "access",
    "label": "Who needs access"
  },
  {
    "key": "presentation",
    "label": "Report presentation"
  },
  {
    "key": "review",
    "label": "Review and activate"
  }
];

export const STATUS_LABELS: Record<CaseStatus, string> = {
  "draft": "Draft",
  "information_requested": "Information requested",
  "awaiting_client": "Awaiting client",
  "in_review": "In review",
  "changes_required": "Changes required",
  "ready_for_approval": "Ready for approval",
  "approved": "Approved",
  "activated": "Activated",
  "withdrawn": "Withdrawn"
};

export const KIND_LABELS: Record<CaseKind, string> = {
  "new_client": "New client",
  "migration": "Legacy migration",
  "amendment": "Amendment"
};

export const INFERENCE_RULES = {
  "version": 1,
  "jurisdiction": {
    "GB": {
      "reporting_currency": "GBP",
      "time_zone": "Europe/London",
      "uk_geography": true
    },
    "IE": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Dublin"
    },
    "SE": {
      "reporting_currency": "SEK",
      "time_zone": "Europe/Stockholm"
    },
    "false": {
      "reporting_currency": "NOK",
      "time_zone": "Europe/Oslo"
    },
    "DK": {
      "reporting_currency": "DKK",
      "time_zone": "Europe/Copenhagen"
    },
    "DE": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Berlin"
    },
    "FR": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Paris"
    },
    "NL": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Amsterdam"
    },
    "ES": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Madrid"
    },
    "IT": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Rome"
    },
    "PT": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Lisbon"
    },
    "BE": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Brussels"
    },
    "AT": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Vienna"
    },
    "FI": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Helsinki"
    },
    "LU": {
      "reporting_currency": "EUR",
      "time_zone": "Europe/Luxembourg"
    },
    "PL": {
      "reporting_currency": "PLN",
      "time_zone": "Europe/Warsaw"
    },
    "CH": {
      "reporting_currency": "CHF",
      "time_zone": "Europe/Zurich"
    },
    "US": {
      "reporting_currency": "USD",
      "time_zone": "America/New_York"
    },
    "CA": {
      "reporting_currency": "CAD",
      "time_zone": "America/Toronto"
    },
    "AU": {
      "reporting_currency": "AUD",
      "time_zone": "Australia/Sydney"
    }
  },
  "jurisdiction_fallback": {
    "time_zone": "UTC"
  },
  "asset_class": {
    "equity_release": {
      "underlying_exposure_type": "RMRT",
      "day_count_convention": "ACT365",
      "payment_frequency": "AT_REDEMPTION"
    },
    "rre": {
      "underlying_exposure_type": "RMRT",
      "day_count_convention": "ACT365",
      "payment_frequency": "MONTHLY"
    },
    "cre": {
      "underlying_exposure_type": "CMRT",
      "day_count_convention": "ACT360",
      "payment_frequency": "QUARTERLY"
    },
    "auto": {
      "underlying_exposure_type": "ALOL",
      "day_count_convention": "ACT365",
      "payment_frequency": "MONTHLY"
    },
    "consumer": {
      "underlying_exposure_type": "CONL",
      "day_count_convention": "ACT365",
      "payment_frequency": "MONTHLY"
    },
    "sme": {
      "underlying_exposure_type": "SMEL",
      "day_count_convention": "ACT360",
      "payment_frequency": "QUARTERLY"
    }
  },
  "risk_retention_holder_by_role": {
    "originator": "ORIG",
    "sponsor": "SPON",
    "original_lender": "OLND",
    "seller": "SELL"
  },
  "cadence": {
    "monthly": "calendar_month_end",
    "weekly": "calendar_month_end",
    "daily": "calendar_month_end",
    "quarterly": "calendar_quarter_end"
  },
  "from_sample": {
    "file_format": {
      "by_extension": {
        "csv": "csv",
        "txt": "csv",
        "xlsx": "xlsx",
        "xls": "xlsx",
        "parquet": "parquet"
      }
    },
    "asset_class_signals_from": "agents.config_bootstrap_agent._ASSET_CLASS_SIGNALS",
    "asset_class_min_hits": 2
  },
  "proposals": {
    "client_id": {
      "from": "client.client_name",
      "transform": "upper_slug",
      "max_length": 12
    },
    "portfolio_id": {
      "pattern": "{portfolio_type}_{sequence:03d}"
    }
  }
} as Record<string, any>;
