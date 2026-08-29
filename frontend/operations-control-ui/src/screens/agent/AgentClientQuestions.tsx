import { useMemo, useState } from "react";
import { useOpsClient } from "@/api/context";
import type { ClientFormField, PackQuestion } from "@/api/agentTypes";
import type { ChecklistRow, InformationRequest } from "@/api/onboardingTypes";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage, useLoad } from "@/lib/useLoad";
import { Empty, Panel } from "./shared";

/**
 * What the client is being asked, what is still outstanding, and the ability
 * to answer it here.
 *
 * The operator could see that a pack contained "22 questions" and could not
 * read them, could not change one, and could not correct a value Trakt had
 * already decided. Every capability needed existed and was tested —
 * `GET /form`, `POST /form`, `POST /steps` — and no screen rendered any of
 * them, so they were reachable only from a test file.
 *
 * This lives in the rail rather than in the timeline, and that is a claim
 * about what it IS. Onboarding is not a linear pass: answers arrive by email,
 * by phone, in a second pack, as a correction three stages later. "What is
 * this client being asked, and what do we already know" is a reference view of
 * the case, not a step in it — so it is reachable at every stage instead of
 * being a step an operator can walk past. It sat inside a timeline stage
 * before, and the predictable thing happened: the stage completed, collapsed,
 * and the only place to record an answer went with it.
 *
 * The outstanding list and the form used to be two panels — Client Onboarding's
 * checklist here, the answerable form somewhere else — so the list of what the
 * client still owes was in one place and the way to satisfy it was in another.
 * They are the same subject and are now one panel.
 *
 * Two populations, and the difference between them is the point:
 *
 *   ASKED     questions the client has not answered. An operator holding the
 *             answers (from a call, an email, a spreadsheet) records them here
 *             on the client's behalf, through the same validated route a
 *             client's own submission takes — authoritative catalogue keys,
 *             checked against the form actually served, written verbatim.
 *
 *   KNOWN     values Trakt already holds. These are NOT re-asked and are not
 *             editable here: a value Trakt derived is corrected by telling it
 *             so in the conversation, where the correction is read, proposed
 *             and confirmed with its provenance recorded. An input box that
 *             silently overwrote a derived value would lose the one thing that
 *             makes the record auditable — who said so, and on what basis.
 *
 * Asking the client is deliberately NOT here. Chasing a client is a step with
 * a place in the sequence, and it stays on the timeline where the sequence is.
 */
export function ClientQuestionsPanel({
  caseId,
  version,
  confirmations,
  checklist,
  requests,
  busy,
  onSaved,
}: {
  caseId: string;
  /** Bumped by the case's own version, so saving refreshes the form. */
  version: number;
  confirmations: PackQuestion[];
  /** Client Onboarding's own list of what the client still owes. */
  checklist: ChecklistRow[];
  requests: InformationRequest[];
  busy: boolean;
  onSaved: () => void;
}) {
  const client = useOpsClient();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [edits, setEdits] = useState<Record<string, unknown>>({});
  const [saving, setSaving] = useState(false);
  const view = useLoad(
    () => (open ? client.getAgentClientForm(caseId) : Promise.resolve(null)),
    [caseId, version, open],
  );

  const form = view.data?.form;
  const changed = Object.keys(edits).length;

  const outstanding = useMemo(() => {
    if (!form) return 0;
    return form.steps.reduce(
      (total, step) =>
        total +
        step.groups.reduce(
          (n, group) => n + group.fields.filter((f) => !present(f.value)).length,
          0,
        ),
      0,
    );
  }, [form]);

  async function save() {
    if (!changed || saving) return;
    setSaving(true);
    try {
      await client.submitAgentClientForm(caseId, edits);
      setEdits({});
      toast.show(copy.agent.questionsSaved, "success");
      onSaved();
    } catch (err) {
      // Shown rather than swallowed: a refused answer names the field and the
      // reason, and that is exactly what the operator needs to correct it.
      toast.show(errorMessage(err), "error");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Panel
      title={copy.agent.questionsHeading}
      action={
        <button
          type="button"
          onClick={() => setOpen((prev) => !prev)}
          className="text-sm font-medium text-blue-700"
        >
          {open ? copy.agent.questionsHide : copy.agent.questionsShow}
        </button>
      }
    >
      <Outstanding checklist={checklist} open={open} />

      {!open ? (
        <p className="mt-2 text-sm text-stone-600">{copy.agent.questionsClosed}</p>
      ) : view.loading ? (
        <Loading />
      ) : view.error ? (
        <ErrorNote message={view.error} onRetry={() => void view.reload()} />
      ) : !form ? (
        <Empty />
      ) : (
        <>
          <p className="text-sm text-stone-600">
            {form.questions} asked of the client, {form.required} of them
            required, {outstanding} still unanswered.
          </p>

          {form.steps.map((step) => (
            <div key={step.key} className="mt-4">
              <h3 className="text-sm font-semibold text-stone-900">{step.label}</h3>
              {step.help && <p className="mt-0.5 text-xs text-stone-500">{step.help}</p>}
              {step.groups.map((group) => (
                <div key={`${group.key}-${group.index ?? 0}`} className="mt-3">
                  {(group.item || step.groups.length > 1) && (
                    <p className="text-xs font-medium uppercase tracking-wide text-stone-400">
                      {group.item || group.label}
                    </p>
                  )}
                  <div className="mt-1 divide-y divide-stone-100">
                    {group.fields.map((field) => (
                      <QuestionRow
                        key={field.key}
                        field={field}
                        value={field.key in edits ? edits[field.key] : field.value}
                        disabled={busy || saving}
                        onChange={(next) =>
                          setEdits((prev) => ({ ...prev, [field.key]: next }))
                        }
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ))}

          {form.locked.length > 0 && (
            <p className="mt-4 text-xs text-stone-500">
              {copy.agent.questionsLocked}{" "}
              {form.locked.map((l) => l.label).join(", ")}.
            </p>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={!changed || busy || saving}
              onClick={() => void save()}
              className="rounded-xl bg-blue-600 px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
            >
              {copy.agent.questionsSave}
            </button>
            {changed > 0 && (
              <button
                type="button"
                disabled={saving}
                onClick={() => setEdits({})}
                className="text-sm font-medium text-stone-600"
              >
                {copy.agent.questionsDiscard}
              </button>
            )}
            <span className="text-xs text-stone-500">
              {changed > 0 ? copy.agent.questionsPending(changed) : copy.agent.questionsHint}
            </span>
          </div>

          {confirmations.length > 0 && (
            <div className="mt-6 border-t border-stone-200 pt-4">
              <h3 className="text-sm font-semibold text-stone-900">
                {copy.agent.questionsKnownHeading}
              </h3>
              <p className="mt-0.5 text-xs text-stone-500">
                {copy.agent.questionsKnownHelp}
              </p>
              <div className="mt-2 divide-y divide-stone-100">
                {confirmations.map((question) => (
                  <div
                    key={`${question.section}.${question.field}.${question.index ?? 0}`}
                    className="flex flex-wrap items-baseline justify-between gap-2 py-1.5"
                  >
                    <span className="text-sm text-stone-700">
                      {question.label}
                      {question.item && (
                        <span className="text-stone-400"> — {question.item}</span>
                      )}
                    </span>
                    <span className="text-sm text-stone-900">{render(question.value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      <Requests requests={requests} />
    </Panel>
  );
}

/**
 * What the client still owes, in Client Onboarding's own words.
 *
 * Shown while the form is closed and withdrawn while it is open: open, the
 * form itself lists every unanswered question, and the same list twice on one
 * screen is noise an operator has to reconcile.
 */
function Outstanding({ checklist, open }: { checklist: ChecklistRow[]; open: boolean }) {
  if (checklist.length === 0) {
    return <p className="text-sm text-stone-500">{copy.agent.checklistEmpty}</p>;
  }
  return (
    <>
      <p className="text-sm text-stone-700">
        {copy.agent.questionsOutstanding(checklist.length)}
      </p>
      {!open && (
        <ul className="mt-1 list-disc space-y-1 pl-4 text-sm text-stone-600">
          {checklist.map((row) => (
            <li key={`${row.section}-${row.field}-${row.index}`}>{row.label}</li>
          ))}
        </ul>
      )}
    </>
  );
}

/** What has been asked so far, and whether it came back. */
function Requests({ requests }: { requests: InformationRequest[] }) {
  if (requests.length === 0) return null;
  return (
    <>
      <p className="mt-4 text-xs uppercase tracking-wide text-stone-400">
        {copy.agent.requestsHeading}
      </p>
      <ul className="mt-1 space-y-1 text-sm text-stone-600">
        {requests.map((request) => (
          <li key={request.request_id} className="flex justify-between gap-2">
            <span>
              {request.items.length} item{request.items.length === 1 ? "" : "s"}
            </span>
            <span className="text-xs text-stone-500">
              {["open", "sent"].includes(request.status)
                ? copy.agent.requestOutstanding
                : copy.agent.requestAnswered}
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}

/** One question, with the control its declared type calls for. */
function QuestionRow({
  field,
  value,
  disabled,
  onChange,
}: {
  field: ClientFormField;
  value: unknown;
  disabled: boolean;
  onChange: (next: unknown) => void;
}) {
  const id = `q-${field.key}`;
  return (
    <div className="py-2">
      <label htmlFor={id} className="block text-sm text-stone-800">
        {field.label}
        <span className="ml-2 text-xs uppercase tracking-wide text-stone-400">
          {field.required ? copy.agent.questionsRequired : copy.agent.questionsOptional}
        </span>
      </label>
      {field.help && <p className="mt-0.5 text-xs text-stone-500">{field.help}</p>}
      <Control id={id} field={field} value={value} disabled={disabled} onChange={onChange} />
    </div>
  );
}

function Control({
  id,
  field,
  value,
  disabled,
  onChange,
}: {
  id: string;
  field: ClientFormField;
  value: unknown;
  disabled: boolean;
  onChange: (next: unknown) => void;
}) {
  const box =
    "mt-1 w-full rounded-lg border border-stone-300 px-2.5 py-1.5 text-sm text-stone-900 disabled:bg-stone-50";

  if (field.options.length > 0) {
    return (
      <select
        id={id}
        className={box}
        disabled={disabled}
        value={String(value ?? "")}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">{copy.agent.questionsUnanswered}</option>
        {field.options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    );
  }
  if (field.type === "boolean" || field.type === "yes_no") {
    return (
      <select
        id={id}
        className={box}
        disabled={disabled}
        value={value === true ? "yes" : value === false ? "no" : ""}
        onChange={(event) =>
          onChange(event.target.value === "" ? null : event.target.value === "yes")
        }
      >
        <option value="">{copy.agent.questionsUnanswered}</option>
        <option value="yes">{copy.agent.yes}</option>
        <option value="no">{copy.agent.no}</option>
      </select>
    );
  }
  if (field.type === "multiline") {
    return (
      <textarea
        id={id}
        rows={3}
        maxLength={field.max_length ?? undefined}
        className={box}
        disabled={disabled}
        value={String(value ?? "")}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  }
  return (
    <input
      id={id}
      type="text"
      maxLength={field.max_length ?? undefined}
      className={box}
      disabled={disabled}
      value={String(value ?? "")}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

function present(value: unknown): boolean {
  if (value === null || value === undefined || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  return true;
}

function render(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? copy.agent.yes : copy.agent.no;
  if (Array.isArray(value)) return value.map((v) => String(v)).join(", ") || "—";
  return String(value);
}
