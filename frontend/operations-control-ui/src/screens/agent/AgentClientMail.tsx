import { useState } from "react";
import { useOpsClient } from "@/api/context";
import type { AgentMailMessage } from "@/api/agentTypes";
import { ErrorNote } from "@/components/ErrorNote";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage } from "@/lib/useLoad";
import { Panel } from "./shared";

/**
 * What the client has sent back, and the decision to take it in.
 *
 * The reader exists on the server and had no caller: an operator could see
 * that a pack had been issued and had no way to see the reply, so a client's
 * loan tape sat in a mailbox and was re-keyed by hand.
 *
 * Three things this screen is careful about, all of them the server's design
 * showing through rather than presentation choices:
 *
 *   NOTHING HAPPENS UNTIL ASKED. Opening the panel does not read the mailbox
 *   and rendering it does not poll. The operator presses a button; that is the
 *   only thing that reads, and it writes nothing and marks nothing read.
 *
 *   EVIDENCE IS SHOWN, NOT HIDDEN. Every matched message says WHY it was tied
 *   to this case — the conversation, the message it answers, the reference in
 *   its subject. An operator taking a client's file onto a case should be able
 *   to see the basis for it without opening an audit trail.
 *
 *   UNMATCHED MAIL IS SHOWN TOO, and cannot be taken in from here. A message
 *   whose only connection is the sender's address is a question for a person:
 *   one contact sits on several onboardings. Hiding it would leave a client's
 *   reply unread; offering a button for it would invite exactly the mistake
 *   the correlator refuses to make.
 */
export function ClientMailPanel({
  caseId,
  busy,
  canRegister,
  onIngested,
}: {
  caseId: string;
  busy: boolean;
  /** Whether the run's state still permits adding a file to this case. A
   *  reply carrying files cannot be taken into a case past that point, and
   *  the server refuses it — so the button is not offered, and the reason is
   *  shown in its place. An action the state machine refuses that renders
   *  anyway reads as a broken button rather than as a rule being enforced. */
  canRegister: boolean;
  onIngested: () => void;
}) {
  const client = useOpsClient();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [checking, setChecking] = useState(false);
  const [taking, setTaking] = useState("");
  const [error, setError] = useState("");
  const [mine, setMine] = useState<AgentMailMessage[] | null>(null);
  const [others, setOthers] = useState<AgentMailMessage[]>([]);
  const [where, setWhere] = useState("");

  async function check() {
    if (checking) return;
    setChecking(true);
    setError("");
    try {
      const found = await client.getAgentMail(caseId);
      setMine(found.mail.for_this_case ?? []);
      setOthers((found.mail.messages ?? []).filter((m) => !m.correlation.matched));
      setWhere(
        found.mail.mailbox
          ? copy.agent.mailMailbox(found.mail.mailbox, found.mail.folder ?? "")
          : (found.mail.note ?? ""),
      );
    } catch (err) {
      // Named rather than swallowed. "The mailbox could not be read" and
      // "nothing has arrived" look identical to an operator otherwise, and
      // they need completely different actions.
      setError(errorMessage(err));
    } finally {
      setChecking(false);
    }
  }

  async function take(message: AgentMailMessage) {
    if (taking) return;
    setTaking(message.graph_id);
    try {
      await client.ingestAgentMail(caseId, [message.graph_id]);
      toast.show(copy.agent.mailIngested, "success");
      onIngested();
      await check();
    } catch (err) {
      toast.show(errorMessage(err), "error");
    } finally {
      setTaking("");
    }
  }

  return (
    <Panel
      title={copy.agent.mailHeading}
      action={
        <button
          type="button"
          onClick={() => {
            const next = !open;
            setOpen(next);
            if (next && mine === null) void check();
          }}
          className="text-sm font-medium text-blue-700"
        >
          {open ? copy.agent.mailHide : copy.agent.mailShow}
        </button>
      }
    >
      {!open ? (
        <p className="text-sm text-stone-600">{copy.agent.mailClosed}</p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={checking || busy}
              onClick={() => void check()}
              className="rounded-xl bg-blue-600 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
            >
              {checking ? copy.agent.mailChecking : copy.agent.mailRecheck}
            </button>
            {where && <span className="text-xs text-stone-500">{where}</span>}
          </div>

          {error && (
            <div className="mt-3">
              <ErrorNote message={error} onRetry={() => void check()} />
            </div>
          )}

          {mine !== null && mine.length === 0 && !error && (
            <p className="mt-3 text-sm text-stone-400">{copy.agent.mailNone}</p>
          )}

          {mine && mine.length > 0 && (
            <ul className="mt-3 space-y-3">
              {mine.map((message) => (
                <MailRow
                  key={message.graph_id}
                  message={message}
                  busy={busy || Boolean(taking)}
                  canRegister={canRegister}
                  taking={taking === message.graph_id}
                  onTake={() => void take(message)}
                />
              ))}
            </ul>
          )}

          {others.length > 0 && (
            <div className="mt-6 border-t border-stone-200 pt-4">
              <h3 className="text-sm font-semibold text-stone-900">
                {copy.agent.mailUnmatchedHeading}
              </h3>
              <p className="mt-0.5 text-xs text-stone-500">
                {copy.agent.mailUnmatchedHelp}
              </p>
              <ul className="mt-2 space-y-2">
                {others.map((message) => (
                  <li
                    key={message.graph_id}
                    className="rounded-xl border border-stone-200 bg-stone-50 px-3 py-2"
                  >
                    <p className="text-sm text-stone-800">
                      {message.subject || copy.agent.mailUnnamed}
                    </p>
                    <p className="text-xs text-stone-500">
                      {copy.agent.mailFrom(message.sender || copy.agent.mailUnnamed)}
                      {message.received_at && ` · ${when(message.received_at)}`}
                    </p>
                    {message.correlation.note && (
                      <p className="mt-1 text-xs text-stone-500">
                        {message.correlation.note}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </Panel>
  );
}

/** One reply that belongs to this case, with the evidence and its files. */
function MailRow({
  message,
  busy,
  canRegister,
  taking,
  onTake,
}: {
  message: AgentMailMessage;
  busy: boolean;
  canRegister: boolean;
  taking: boolean;
  onTake: () => void;
}) {
  const files = message.attachments.filter((a) => !a.inline);
  // Precise, not blanket: a reply with no files needs nothing registered, so
  // its words can still be recorded on a case that has moved on.
  const blocked = files.length > 0 && !canRegister;
  return (
    <li className="rounded-xl border border-stone-200 bg-white px-3 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium text-stone-900">
          {message.subject || copy.agent.mailUnnamed}
        </p>
        {message.received_at && (
          <span className="text-xs text-stone-500">{when(message.received_at)}</span>
        )}
      </div>
      <p className="text-xs text-stone-500">
        {copy.agent.mailFrom(message.sender_name || message.sender || copy.agent.mailUnnamed)}
      </p>

      {message.body_text && (
        <p className="mt-2 whitespace-pre-wrap rounded-lg bg-stone-50 px-2.5 py-2 text-sm text-stone-700">
          {message.body_text}
        </p>
      )}

      <p className="mt-2 text-xs text-stone-500">
        {files.length === 0
          ? copy.agent.mailNoAttachments
          : files
              .map((f) =>
                f.oversize
                  ? `${f.name} — ${copy.agent.mailOversize}`
                  : !f.readable
                    ? `${f.name} — ${copy.agent.mailUnreadable}`
                    : f.name,
              )
              .join(", ")}
      </p>

      {message.correlation.bases.length > 0 && (
        <p className="mt-1 text-xs text-stone-500">
          {copy.agent.mailMatchedOn(message.correlation.bases)}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-3">
        {message.already_ingested ? (
          <span className="text-sm font-medium text-stone-500">
            {copy.agent.mailAlready}
          </span>
        ) : blocked ? (
          <span className="text-xs text-stone-500">
            {copy.agent.mailCannotRegister}
          </span>
        ) : (
          <button
            type="button"
            disabled={busy}
            onClick={onTake}
            className="rounded-xl bg-stone-900 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
          >
            {taking ? copy.agent.mailTaking : copy.agent.mailTake}
          </button>
        )}
      </div>

      {!message.already_ingested && !blocked && (
        <p className="mt-2 text-xs text-stone-500">{copy.agent.mailNotApplied}</p>
      )}
    </li>
  );
}

/** A timestamp a person reads, or the raw value when it is not one. */
function when(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}
