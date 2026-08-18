"use client";

import { useEffect, useId, useRef, useState } from "react";

import { buttonStyles, cx } from "@/components/ui";
import { readAttribution, track } from "@/lib/analytics";
import {
  LEAD_FIELD_LIMITS,
  validateLead,
  type LeadErrors,
} from "@/lib/lead-validation";

type Status = "idle" | "submitting" | "success" | "error";

/**
 * The lead form.
 *
 * Client-side validation mirrors the server's for immediate feedback; the
 * server re-validates and is the authority. Two bot controls are carried here
 * and enforced server-side: a honeypot field (hidden from sighted users and
 * from assistive technology) and the elapsed time since the form mounted.
 */
export function LeadForm() {
  const [status, setStatus] = useState<Status>("idle");
  const [errors, setErrors] = useState<LeadErrors>({});
  const [message, setMessage] = useState<string | null>(null);
  // Set in an effect rather than during render — render must stay pure.
  const mountedAt = useRef<number>(0);
  useEffect(() => {
    mountedAt.current = Date.now();
  }, []);
  const ids = {
    name: useId(),
    email: useId(),
    company: useId(),
    role: useId(),
    message: useId(),
    consent: useId(),
    honeypot: useId(),
  };

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (status === "submitting") return;

    const form = new FormData(event.currentTarget);
    const payload = {
      name: String(form.get("name") ?? ""),
      email: String(form.get("email") ?? ""),
      company: String(form.get("company") ?? ""),
      role: String(form.get("role") ?? ""),
      message: String(form.get("message") ?? ""),
      consent: form.get("consent") === "on",
      website: String(form.get("website") ?? ""),
      elapsedMs: Date.now() - mountedAt.current,
      // Re-sanitised server-side; the browser is not trusted for this.
      attribution: readAttribution(),
    };

    const local = validateLead(payload);
    if (!local.ok) {
      setErrors(local.errors);
      setStatus("error");
      setMessage(null);
      return;
    }

    setErrors({});
    setStatus("submitting");

    try {
      const response = await fetch("/api/leads", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = (await response.json().catch(() => ({}))) as {
        status?: string;
        errors?: LeadErrors;
        message?: string;
      };

      if (response.ok) {
        setStatus("success");
        track("lead_submit_success");
        return;
      }

      if (data.errors) setErrors(data.errors);
      setStatus("error");
      setMessage(
        data.message ??
          (response.status === 429
            ? "Too many requests. Please wait a moment and try again."
            : "Please check the highlighted fields and try again."),
      );
      track("lead_submit_failure", { outcome: String(response.status) });
    } catch {
      setStatus("error");
      setMessage("We could not reach the server. Please try again.");
      track("lead_submit_failure", { outcome: "network" });
    }
  }

  if (status === "success") {
    return (
      <div
        role="status"
        // A system-state context: the submission succeeded. Read by the e2e
        // colour guard, which otherwise forbids mint outside a state.
        data-state-colour="lead-success"
        className="rounded-xl border border-mint-400/35 bg-mint-400/[0.07] p-6"
      >
        <h3 className="text-subhead font-semibold text-ink-100">Thank you — that&apos;s with us.</h3>
        <p className="mt-2 text-body leading-relaxed text-ink-300">
          We will be in touch to arrange a walkthrough against your own portfolio data,
          reporting requirements and Microsoft 365 environment.
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={onSubmit} noValidate className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          id={ids.name}
          name="name"
          label="Name"
          autoComplete="name"
          maxLength={LEAD_FIELD_LIMITS.name}
          error={errors.name}
          required
        />
        <Field
          id={ids.email}
          name="email"
          type="email"
          label="Work email"
          autoComplete="email"
          maxLength={LEAD_FIELD_LIMITS.email}
          error={errors.email}
          required
        />
        <Field
          id={ids.company}
          name="company"
          label="Company"
          autoComplete="organization"
          maxLength={LEAD_FIELD_LIMITS.company}
          error={errors.company}
          required
        />
        <Field
          id={ids.role}
          name="role"
          label="Role"
          optional
          autoComplete="organization-title"
          maxLength={LEAD_FIELD_LIMITS.role}
          error={errors.role}
        />
      </div>

      <div>
        <label htmlFor={ids.message} className="block text-small font-medium text-ink-300">
          Message <span className="text-ink-500">(optional)</span>
        </label>
        <textarea
          id={ids.message}
          name="message"
          rows={3}
          maxLength={LEAD_FIELD_LIMITS.message}
          className="mt-1.5 w-full rounded-lg border border-line bg-navy-950 px-3.5 py-2.5 text-body text-ink-100 placeholder:text-ink-500"
          placeholder="Asset class, portfolio size, reporting requirements…"
        />
      </div>

      {/* Honeypot. Hidden from sighted users and from assistive technology. */}
      <div aria-hidden="true" className="absolute h-0 w-0 overflow-hidden opacity-0">
        <label htmlFor={ids.honeypot}>Website</label>
        <input id={ids.honeypot} name="website" type="text" tabIndex={-1} autoComplete="off" />
      </div>

      <div>
        <label htmlFor={ids.consent} className="flex items-start gap-2.5 text-small leading-relaxed text-ink-300">
          <input
            id={ids.consent}
            name="consent"
            type="checkbox"
            required
            aria-invalid={errors.consent ? true : undefined}
            aria-describedby={errors.consent ? `${ids.consent}-error` : undefined}
            className="mt-0.5 h-4 w-4 shrink-0 rounded border-line bg-navy-950 accent-peri-400"
          />
          <span>
            I am happy for Trakt to contact me about this enquiry. We use your details
            only to respond, and do not add you to a marketing list.
          </span>
        </label>
        {errors.consent ? (
          <p id={`${ids.consent}-error`} className="mt-1.5 text-small text-rose-400">
            {errors.consent}
          </p>
        ) : null}
      </div>

      {message ? (
        <p role="alert" className="rounded-lg border border-rose-400/35 bg-rose-400/10 px-3 py-2 text-body text-rose-400">
          {message}
        </p>
      ) : null}

      <button type="submit" disabled={status === "submitting"} className={cx(buttonStyles.primary, "w-full sm:w-auto")}>
        {status === "submitting" ? "Sending…" : "Book a tailored demonstration"}
      </button>
    </form>
  );
}

function Field({
  id,
  name,
  label,
  error,
  type = "text",
  required,
  optional,
  autoComplete,
  maxLength,
}: {
  id: string;
  name: string;
  label: string;
  error?: string;
  type?: string;
  required?: boolean;
  optional?: boolean;
  autoComplete?: string;
  maxLength?: number;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-small font-medium text-ink-300">
        {label}
        {required ? <span aria-hidden="true" className="ml-0.5 text-peri-400">*</span> : null}
        {optional ? <span className="text-ink-500"> (optional)</span> : null}
      </label>
      <input
        id={id}
        name={name}
        type={type}
        required={required}
        autoComplete={autoComplete}
        maxLength={maxLength}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${id}-error` : undefined}
        className={cx(
          "mt-1.5 w-full rounded-lg border bg-navy-950 px-3.5 py-2.5 text-body text-ink-100 placeholder:text-ink-500",
          error ? "border-rose-400/60" : "border-line",
        )}
      />
      {error ? (
        <p id={`${id}-error`} className="mt-1.5 text-small text-rose-400">
          {error}
        </p>
      ) : null}
    </div>
  );
}
