import { useState } from "react";
import { DialogButtons, Modal } from "@/components/admin/primitives";
import { copy } from "@/lib/copy";
import { Field, TextInput } from "./primitives";

/**
 * Explicit confirmation before abandoning a case.
 *
 * The reason is required rather than optional, because a cancelled onboarding
 * is kept and read later — by whoever asks in six months why this client was
 * started and never finished. "Cancelled by Operator" alone does not answer
 * that.
 *
 * The dialog says what is *not* happening, which is the thing an operator is
 * actually anxious about: on a new client nothing exists yet, and on an
 * amendment the version in force is untouched.
 */
export function WithdrawDialog({
  kind,
  busy,
  onCancel,
  onConfirm,
}: {
  kind: string;
  busy?: boolean;
  onCancel: () => void;
  onConfirm: (reason: string) => void;
}) {
  const [reason, setReason] = useState("");
  return (
    <Modal labelledBy="withdraw-heading">
      <h2 id="withdraw-heading" className="text-lg font-semibold text-stone-900">
        {copy.onboarding.withdrawHeading}
      </h2>
      <p className="mt-3 text-sm leading-relaxed text-stone-600">
        {kind === "amendment"
          ? copy.onboarding.withdrawExplainAmendment
          : copy.onboarding.withdrawExplain}
      </p>
      <div className="mt-5">
        <Field label={copy.onboarding.withdrawReason} required>
          <TextInput
            value={reason}
            onChange={setReason}
            placeholder={copy.onboarding.reasonPlaceholder}
          />
        </Field>
      </div>
      <DialogButtons
        onCancel={onCancel}
        onConfirm={() => onConfirm(reason.trim())}
        confirmLabel={copy.onboarding.withdrawConfirm}
        busy={busy}
        destructive
        disabled={!reason.trim()}
      />
    </Modal>
  );
}
