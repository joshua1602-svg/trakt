# Trakt Operations Control Centre — Phase 1 design pack

Phase 1 deliverables for the Trakt Operations Control Centre: a new governed
operational layer that sits **above** the existing agents. No implementation is
included in this phase — these documents are the proposal that precedes the
approval checkpoint.

| Doc | Deliverable |
|-----|-------------|
| [01_architecture_proposal.md](01_architecture_proposal.md) | Architecture proposal — layers, seams, non-invasive integration strategy |
| [02_react_information_architecture.md](02_react_information_architecture.md) | React information architecture — screens, navigation, UX language |
| [03_workflow_diagrams.md](03_workflow_diagrams.md) | Workflow diagrams — the three operational workflows end to end |
| [04_state_diagrams.md](04_state_diagrams.md) | State diagrams — workflow, stage, review item, rule, publication |
| [05_api_design.md](05_api_design.md) | Operations Control API design |
| [06_persistence_design.md](06_persistence_design.md) | Persistence design — `operations-control` blob container layout |
| [07_governed_agent_result.md](07_governed_agent_result.md) | Governed Agent Result contract |
| [08_operational_rule_model.md](08_operational_rule_model.md) | Operational rule model — scoped, versioned, persistent decisions |
| [09_migration_impact_assessment.md](09_migration_impact_assessment.md) | Migration impact assessment + approval checkpoint |

## Design principles (apply to every document)

1. **Less is more.** Show only what is needed to make the next operational decision.
2. **The operator is not an engineer.** No JSON, stack traces, blob paths,
   container names, schema IDs or Python errors ever reach the screen.
3. **Do not modify the existing pipelines.** The Control Centre wraps existing
   agents through adapters and reads their manifests; business calculations stay
   inside the agents.
4. **Every approval becomes a governed, scoped, versioned rule.**
5. **Nothing publishes automatically.** Publication is always an explicit,
   audited human approval.
