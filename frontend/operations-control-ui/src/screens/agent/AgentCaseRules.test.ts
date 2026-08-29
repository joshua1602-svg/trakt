import { describe, expect, it } from "vitest";
import type { CaseProblem } from "@/api/onboardingTypes";
import { operatorBlocking } from "./AgentCase";

/**
 * The ownership split behind "What Trakt still needs from you".
 *
 * The panel used to render every blocking problem, which meant the client's
 * eleven outstanding fields appeared twice on one screen — once as a checklist
 * under the client questions, once as sentences here — and anything genuinely
 * the operator's was buried among them.
 */
function problem(field: string, owner: "client" | "operator"): CaseProblem {
  return {
    section: "identity",
    field,
    message: `${field} is needed.`,
    severity: "blocking",
    index: null,
    owner,
  };
}

describe("what is the operator's to solve", () => {
  it("keeps only what is not the client's to answer", () => {
    const kept = operatorBlocking([
      problem("legal_entity_identifier", "client"),
      problem("pipeline_contract", "operator"),
      problem("reporting_contact_email", "client"),
    ]);
    expect(kept.map((p) => p.field)).toEqual(["pipeline_contract"]);
  });

  it("empties when everything outstanding is the client's", () => {
    expect(operatorBlocking([problem("a", "client"), problem("b", "client")]))
      .toEqual([]);
  });

  /**
   * The rule reads the server's own `owner`, which comes from whether the
   * catalogue asks that field of a client at all. Deriving it here from
   * checklist membership would have been wrong in a way that matters:
   * `client_checklist` EXCLUDES anything sitting in an open request, so
   * pressing "ask the client" empties it — and the client's items would tip
   * into this panel at the exact moment it became most true that we were
   * waiting on them. Ownership does not depend on what has been asked.
   */
  it("does not change when the client has been asked", () => {
    const rows = [problem("lei", "client"), problem("pipeline", "operator")];
    const before = operatorBlocking(rows);
    // Nothing about asking alters a problem's owner, so nothing here moves.
    expect(operatorBlocking(rows)).toEqual(before);
    expect(before.map((p) => p.field)).toEqual(["pipeline"]);
  });

  it("treats an unstamped problem as the operator's rather than hiding it", () => {
    // Degrading to "not shown" would lose a blocker silently; degrading to
    // "yours" merely shows one row too many.
    const orphan = { ...problem("x", "operator"), owner: undefined as never };
    expect(operatorBlocking([orphan])).toHaveLength(1);
  });
});
