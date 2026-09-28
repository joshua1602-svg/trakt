import { useState } from "react";
import { Modal } from "@/components/admin/primitives";
import { copy } from "@/lib/copy";

/**
 * Explicit confirmation before ending an Agent case, and the reason it ended.
 *
 * WHY THE REASON IS REQUIRED
 *
 * Cancelling is terminal — there is no transition out of it. The record is
 * kept and read later by whoever asks why this client was started and never
 * finished, and the one answer that cannot help them is a default sentence.
 * Until this dialog existed the only route was typing the instruction into the
 * conversation, which accepted whatever was said: an operator who typed
 * "Cancel this case." got exactly that back as the reason on the record.
 *
 * So the confirm button stays disabled until something is written, matching
 * Client Onboarding's dialog for the identical act.
 *
 * WHY THE BUTTONS ARE NOT `DialogButtons`
 *
 * That primitive labels its dismiss button "Cancel". In a dialog about
 * CANCELLING A CASE, two buttons reading "Cancel" and "Cancel this case" is a
 * coin toss on the one screen where being wrong is irreversible. The dismiss
 * is named for what it does instead — "Keep working on it" — so neither button
 * can be read as the other.
 */
export function AgentCancelDialog({
  live,
  busy,
  onDismiss,
  onConfirm,
}: {
  /** A real onboarding rather than a rehearsal, which changes only the
   *  reassurance text: neither has activated, so neither removes anything. */
  live: boolean;
  busy?: boolean;
  onDismiss: () => void;
  onConfirm: (reason: string) => void;
}) {
  const [reason, setReason] = useState("");
  const ready = reason.trim().length > 0 && !busy;

  return (
    <Modal labelledBy="agent-cancel-heading">
      <h2
        id="agent-cancel-heading"
        className="text-lg font-semibold text-stone-900"
      >
        {copy.agent.cancelHeading}
      </h2>
      <p className="mt-3 text-sm leading-relaxed text-stone-600">
        {live ? copy.agent.cancelExplainLive : copy.agent.cancelExplain}
      </p>

      <div className="mt-5">
        <label
          htmlFor="agent-cancel-reason"
          className="block text-sm font-medium text-stone-900"
        >
          {copy.agent.cancelReason}
          <span aria-hidden className="ml-1 text-rose-600">
            *
          </span>
        </label>
        <p id="agent-cancel-reason-help" className="mt-1 text-xs text-stone-500">
          {copy.agent.cancelReasonHelp}
        </p>
        {/* A textarea, not a single-line input: the useful answer is a
            sentence, and a box the size of the answer is how a form asks for
            one. */}
        <textarea
          id="agent-cancel-reason"
          aria-describedby="agent-cancel-reason-help"
          required
          rows={3}
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          placeholder={copy.agent.cancelReasonPlaceholder}
          className="mt-2 w-full rounded-xl border border-stone-300 px-3 py-2 text-sm text-stone-900 placeholder:text-stone-400 focus:border-stone-500 focus:outline-none"
        />
      </div>

      <div className="mt-6 flex justify-end gap-3">
        <button
          type="button"
          onClick={onDismiss}
          className="rounded-xl border border-stone-300 bg-white px-4 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50"
        >
          {copy.agent.cancelKeep}
        </button>
        <button
          type="button"
          disabled={!ready}
          onClick={() => onConfirm(reason.trim())}
          className="rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-700 disabled:opacity-50"
        >
          {copy.agent.cancelConfirm}
        </button>
      </div>
    </Modal>
  );
}
