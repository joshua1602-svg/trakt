/**
 * A decision is raised either against a workflow run or against an input pack
 * that has not opened one yet (a file still to be identified). The two carry
 * different identifiers, and only the first has a case file to open, so the
 * queue and the deep-link route both have to tell them apart.
 */
export function isWorkflowId(id: string): boolean {
  return /^wf[-_]/.test(id ?? "");
}

/** Where a question should be answered. */
export function decisionHref(decisionId: string, workflowId: string): string {
  return isWorkflowId(workflowId)
    ? `/workflows/${encodeURIComponent(workflowId)}?decision=${encodeURIComponent(decisionId)}`
    : `/reviews/${encodeURIComponent(decisionId)}`;
}
