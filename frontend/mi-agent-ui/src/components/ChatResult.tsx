import type { Artifact } from "@/domain";
import { isChartArtifact, isTableArtifact } from "@/domain";

/**
 * What the conversation says about a result that is ALREADY ON SCREEN.
 *
 * The chat rail and the Artifact Workspace render side by side, and a new
 * artifact expands the workspace, scrolls to itself and flash-highlights (see
 * AppShell's auto-reveal). So this used to carry two things that had no work
 * left to do:
 *
 *   * key-number chips — "Groups: 52", "Coverage: 100%". Both are already in
 *     the answer's execution receipt and the artifact's reconciliation footer,
 *     so the reader met the same figure two or three times in three formats
 *     with nothing to say which was authoritative.
 *   * "Open chart / Open table in workspace" buttons — navigation to something
 *     already open, already scrolled to, already glowing.
 *
 * The one case that DOES need saying is the opposite: the workspace was
 * cleared, so the result the answer refers to is no longer there. That is a
 * sentence, not a control.
 *
 * The rule this holds: the chat states, the workspace shows.
 */
export function ChatResult({
  artifacts,
  workspaceArtifactIds,
}: {
  artifacts: Artifact[];
  /** Retained for signature compatibility; pinning happens in the workspace. */
  onTogglePin?: (id: string) => void;
  /** Retained for signature compatibility; the workspace reveals itself. */
  onOpenArtifact?: (id: string) => void;
  /** Ids still present in the workspace. Absent means "don't know" — say nothing. */
  workspaceArtifactIds?: Set<string>;
}) {
  if (artifacts.length === 0 || !workspaceArtifactIds) return null;

  // Only the renderable results are worth mentioning; a validation artifact
  // being cleared is not something the reader needs to act on.
  const shown = artifacts.filter((a) => isChartArtifact(a) || isTableArtifact(a));
  if (shown.length === 0 || shown.some((a) => workspaceArtifactIds.has(a.id))) return null;

  return (
    <div className="mt-2 text-[11px] text-ink-500" data-testid="chat-result-cleared">
      This result is no longer in the workspace — ask again to regenerate it.
    </div>
  );
}
