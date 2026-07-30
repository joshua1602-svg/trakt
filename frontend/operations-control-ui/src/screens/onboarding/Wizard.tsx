import { useCallback, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Check, Plus, Trash2 } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type {
  OnboardingDraft,
  OnboardingProblem,
  OnboardingProfile,
  OnboardingReference,
  OnboardingReview,
  OnboardingStep,
  PortfolioStanding,
  StandingField,
} from "@/api/onboardingTypes";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import {
  ActionChip,
  Card,
  Field,
  KeyValue,
  Note,
  PrimaryButton,
  SecondaryButton,
  SectionHeading,
  Select,
  TextInput,
  Toggle,
} from "@/components/onboarding/primitives";
import { Page } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage, useLoad } from "@/lib/useLoad";

const label = (value: string) =>
  value.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

function problemFor(problems: OnboardingProblem[], field: string): string | undefined {
  return problems.find((p) => p.field === field)?.message;
}

function options(values: string[]) {
  return values.map((v) => ({ value: v, label: label(v) }));
}

/** The step rail. Numbers are the operator's map of the task. */
function StepRail({
  steps,
  current,
  onSelect,
}: {
  steps: OnboardingDraft["steps"];
  current: OnboardingStep;
  onSelect: (step: OnboardingStep) => void;
}) {
  return (
    <ol className="mb-6 flex flex-wrap gap-2">
      {steps.map((step, index) => {
        const active = step.key === current;
        return (
          <li key={step.key}>
            <button
              type="button"
              onClick={() => onSelect(step.key)}
              className={clsx(
                "flex items-center gap-2 rounded-xl border px-3 py-2 text-sm transition-colors",
                active
                  ? "border-stone-900 bg-stone-900 text-white"
                  : "border-stone-200 bg-white text-stone-600 hover:border-stone-400",
              )}
            >
              <span
                className={clsx(
                  "flex h-5 w-5 items-center justify-center rounded-full text-xs font-semibold",
                  active ? "bg-white text-stone-900" : "bg-stone-100 text-stone-500",
                )}
              >
                {step.problems === 0 ? <Check className="h-3 w-3" aria-hidden /> : index + 1}
              </span>
              {step.label}
            </button>
          </li>
        );
      })}
    </ol>
  );
}

// --------------------------------------------------------------------------- //
// Step 1 — Client
// --------------------------------------------------------------------------- //

function ClientStep({
  profile,
  problems,
  provenance,
  onChange,
}: {
  profile: OnboardingProfile;
  problems: OnboardingProblem[];
  provenance: Record<string, string>;
  onChange: (patch: Record<string, string>) => void;
}) {
  const c = profile.client;
  const set = (key: string) => (value: string) => onChange({ [key]: value });
  const field = (key: keyof typeof c, text: string, hint?: string, required = true) => (
    <Field
      label={text}
      required={required}
      hint={hint}
      origin={provenance[`client.${key}`]}
      problem={problemFor(problems, key)}
    >
      <TextInput
        value={String(c[key] ?? "")}
        onChange={set(key)}
        invalid={Boolean(problemFor(problems, key))}
      />
    </Field>
  );

  return (
    <div className="space-y-6">
      <Card title="Who is the client?">
        <div className="grid gap-4 sm:grid-cols-2">
          {field("client_id", "Client identifier", "Short and stable. Used everywhere Trakt refers to this client.")}
          {field("client_name", "Client name")}
          {field("legal_entity_name", "Legal entity name", "The name held against the identifier in the global register.")}
          {field("lei", "Legal entity identifier", "Twenty characters.")}
          {field("jurisdiction", "Jurisdiction", "Two-letter country code, for example GB.")}
          {field("reporting_currency", "Reporting currency", "Three-letter code, for example GBP.")}
          {field("time_zone", "Reporting time zone")}
        </div>
      </Card>

      <Card title="Who does Trakt talk to?">
        <div className="grid gap-4 sm:grid-cols-2">
          {field("primary_reporting_contact", "Primary reporting contact")}
          {field("reporting_email", "Reporting email")}
          {field("operational_contact", "Operational contact")}
          {field("operational_email", "Operational email")}
        </div>
      </Card>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Step 2 — Portfolios
// --------------------------------------------------------------------------- //

function emptyPortfolio(assetClass: string): PortfolioStanding {
  return {
    portfolio_id: "",
    display_name: "",
    portfolio_type: "direct",
    asset_class: assetClass,
    structure: "on_balance_sheet",
    funded_cadence: "monthly",
    pipeline_cadence: "",
    originates: null,
    warehouse_lender: "",
    source_system: "",
  };
}

function PortfolioStep({
  profile,
  reference,
  problems,
  onChange,
}: {
  profile: OnboardingProfile;
  reference: OnboardingReference;
  problems: OnboardingProblem[];
  onChange: (portfolios: PortfolioStanding[]) => void;
}) {
  const vocab = reference.vocabularies;
  const assetOptions = vocab.asset_classes.map((a) => ({ value: a.value, label: a.label }));

  function update(index: number, patch: Partial<PortfolioStanding>) {
    onChange(profile.portfolios.map((p, i) => (i === index ? { ...p, ...patch } : p)));
  }

  return (
    <div className="space-y-6">
      <Note>
        Every choice below is one Trakt already understands — nothing here is free text where a
        governed value exists.
      </Note>

      {profile.portfolios.map((portfolio, index) => (
        <Card
          key={index}
          title={portfolio.display_name || `Portfolio ${index + 1}`}
          actions={
            <button
              type="button"
              onClick={() => onChange(profile.portfolios.filter((_, i) => i !== index))}
              className="flex items-center gap-1 text-sm text-stone-500 hover:text-red-600"
            >
              <Trash2 className="h-4 w-4" aria-hidden />
              Remove
            </button>
          }
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Portfolio identifier" required>
              <TextInput
                value={portfolio.portfolio_id}
                onChange={(v) => update(index, { portfolio_id: v })}
              />
            </Field>
            <Field label="Display name" required>
              <TextInput
                value={portfolio.display_name}
                onChange={(v) => update(index, { display_name: v })}
              />
            </Field>
            <Field label="Portfolio type" required hint="Written by this client, or bought in.">
              <Select
                value={portfolio.portfolio_type}
                onChange={(v) => update(index, { portfolio_type: v })}
                options={options(vocab.portfolio_types)}
              />
            </Field>
            <Field label="Asset class" required>
              <Select
                value={portfolio.asset_class}
                onChange={(v) => update(index, { asset_class: v })}
                options={assetOptions}
                placeholder="Choose an asset class"
              />
            </Field>
            <Field label="How the book is held" required>
              <Select
                value={portfolio.structure}
                onChange={(v) => update(index, { structure: v })}
                options={options(vocab.portfolio_structures)}
              />
            </Field>
            <Field label="Who supplies the data">
              <TextInput
                value={portfolio.source_system}
                onChange={(v) => update(index, { source_system: v })}
              />
            </Field>
            <Field label="Funded reporting cadence" required>
              <Select
                value={portfolio.funded_cadence}
                onChange={(v) => update(index, { funded_cadence: v })}
                options={options(vocab.cadences)}
              />
            </Field>
            <Field
              label="Pipeline reporting cadence"
              hint="Leave blank where the book sends no pipeline data."
            >
              <Select
                value={portfolio.pipeline_cadence}
                onChange={(v) => update(index, { pipeline_cadence: v })}
                options={options(vocab.cadences)}
                placeholder="No pipeline data"
              />
            </Field>
          </div>
        </Card>
      ))}

      {problems.length > 0 && (
        <Note tone="warn">{problems.map((p) => p.message).join(" ")}</Note>
      )}

      <SecondaryButton
        onClick={() =>
          onChange([...profile.portfolios, emptyPortfolio(vocab.asset_classes[0]?.value ?? "")])
        }
      >
        <span className="flex items-center gap-2">
          <Plus className="h-4 w-4" aria-hidden />
          Add a portfolio
        </span>
      </SecondaryButton>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Step 3 — Reporting
// --------------------------------------------------------------------------- //

function ReportingStep({
  draft,
  reference,
  onChange,
}: {
  draft: OnboardingDraft;
  reference: OnboardingReference;
  onChange: (products: string[]) => void;
}) {
  const chosen = draft.profile.reporting.products;
  return (
    <div className="space-y-6">
      <Note>
        Trakt works out what a client is eligible for from the books they have. You choose which
        of those they actually receive.
      </Note>
      <Card title="Reporting products">
        <div className="space-y-3">
          {reference.vocabularies.products.map((product) => {
            const derived = draft.derived_reporting[product.key];
            const alwaysOn = product.always_applies;
            const eligible = derived?.eligible ?? true;
            return (
              <Toggle
                key={product.key}
                label={product.label}
                description={
                  alwaysOn
                    ? `Always prepared. ${product.derived_from}`
                    : eligible
                      ? derived?.reason
                      : derived?.reason ?? "Not available for these portfolios."
                }
                checked={alwaysOn || chosen.includes(product.key)}
                disabled={alwaysOn || !eligible}
                onChange={(on) =>
                  onChange(
                    on
                      ? [...chosen, product.key]
                      : chosen.filter((key) => key !== product.key),
                  )
                }
              />
            );
          })}
        </div>
      </Card>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Step 4 — Regime configuration
// --------------------------------------------------------------------------- //

function RegimeField({
  field,
  value,
  problem,
  origin,
  onChange,
}: {
  field: StandingField;
  value: string | boolean | undefined;
  problem?: string;
  origin?: string;
  onChange: (value: string | boolean) => void;
}) {
  const text = value === undefined || value === null ? "" : String(value);
  const hint = [field.help, field.regime_code ? `Reported as ${field.regime_code}.` : ""]
    .filter(Boolean)
    .join(" ");

  if (field.format === "boolean") {
    return (
      <Toggle
        label={field.label}
        description={field.help}
        checked={value === true || text === "true"}
        onChange={onChange}
      />
    );
  }
  if (field.format === "enum" && field.options) {
    const opts = field.options.map((o) =>
      typeof o === "string" ? { value: o, label: o } : { value: o.value, label: o.label },
    );
    return (
      <Field label={field.label} required={field.required} hint={hint} problem={problem} origin={origin}>
        <Select value={text} onChange={onChange} options={opts} placeholder="Choose a value" />
      </Field>
    );
  }
  if (field.format === "yes_no") {
    return (
      <Field label={field.label} required={field.required} hint={hint} problem={problem} origin={origin}>
        <Select
          value={text}
          onChange={onChange}
          options={[
            { value: "Y", label: "Yes" },
            { value: "N", label: "No" },
          ]}
          placeholder="Choose"
        />
      </Field>
    );
  }
  return (
    <Field label={field.label} required={field.required} hint={hint} problem={problem} origin={origin}>
      <TextInput value={text} onChange={onChange} invalid={Boolean(problem)} />
    </Field>
  );
}

function RegimeStep({
  draft,
  reference,
  onChange,
}: {
  draft: OnboardingDraft;
  reference: OnboardingReference;
  onChange: (product: string, key: string, value: string | boolean) => void;
}) {
  const products = draft.profile.reporting.products.filter(
    (key) => (reference.standing_fields.products[key]?.fields ?? []).length > 0,
  );

  if (products.length === 0) {
    return (
      <Note>
        The reporting products chosen so far need no standing regime configuration. Management
        information is prepared from the data itself.
      </Note>
    );
  }

  return (
    <div className="space-y-6">
      <Note>
        These values stay the same from period to period, so Trakt asks once and reuses them on
        every delivery.
      </Note>
      {products.map((key) => {
        const product = reference.standing_fields.products[key];
        const answers = draft.profile.regime[key] ?? {};
        return (
          <Card key={key} title={product.label}>
            <div className="grid gap-4 sm:grid-cols-2">
              {product.fields.map((field) => (
                <RegimeField
                  key={field.key}
                  field={field}
                  value={answers[field.key]}
                  problem={problemFor(draft.problems, `${key}.${field.key}`)}
                  origin={draft.provenance[`regime.${key}.${field.key}`]}
                  onChange={(value) => onChange(key, field.key, value)}
                />
              ))}
            </div>
            {(product.unrepresented ?? []).length > 0 && (
              <div className="mt-6 border-t border-stone-100 pt-4">
                <SectionHeading>Not held here</SectionHeading>
                <ul className="space-y-2">
                  {(product.unrepresented ?? []).map((entry) => (
                    <li key={entry.key} className="text-sm text-stone-500">
                      <span className="font-medium text-stone-700">{entry.label}</span> —{" "}
                      {entry.reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Step 5 — Investor and static reporting
// --------------------------------------------------------------------------- //

const STATIC_FIELDS: { key: string; label: string; hint?: string }[] = [
  { key: "app_title", label: "Report title" },
  { key: "primary_color", label: "Brand colour", hint: "For example #0B1F3B." },
  { key: "logo_uri", label: "Logo" },
  { key: "disclaimer", label: "Disclaimer" },
  { key: "investor_contact", label: "Investor contact" },
  { key: "investor_email", label: "Investor email" },
  { key: "trustee_name", label: "Trustee" },
  { key: "warehouse_lender", label: "Warehouse lender" },
  { key: "payment_convention", label: "Payment convention", hint: "For example QUARTERLY." },
  { key: "day_count_convention", label: "Day-count convention", hint: "For example ACT365." },
  { key: "reporting_convention", label: "Reporting convention" },
];

function StaticReportingStep({
  profile,
  provenance,
  onChange,
}: {
  profile: OnboardingProfile;
  provenance: Record<string, string>;
  onChange: (patch: Record<string, string>) => void;
}) {
  const sr = profile.static_reporting as unknown as Record<string, string>;
  return (
    <div className="space-y-6">
      <Note>
        Only what stays the same every period belongs here. Anything that changes with the
        reporting date is asked for at delivery time.
      </Note>
      <Card title="How reports look and who they name">
        <div className="grid gap-4 sm:grid-cols-2">
          {STATIC_FIELDS.map(({ key, label: text, hint }) => (
            <Field
              key={key}
              label={text}
              hint={hint}
              origin={
                key === "app_title" || key === "primary_color"
                  ? provenance["static_reporting.branding"]
                  : key.endsWith("convention")
                    ? provenance["static_reporting.conventions"]
                    : undefined
              }
            >
              <TextInput value={sr[key] ?? ""} onChange={(v) => onChange({ [key]: v })} />
            </Field>
          ))}
        </div>
      </Card>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Step 6 — Source registration
// --------------------------------------------------------------------------- //

function SourcesStep({ draft }: { draft: OnboardingDraft }) {
  const sources = draft.profile.sources;
  return (
    <div className="space-y-6">
      <Note>
        These follow from the portfolios and reporting you have already answered. Trakt creates
        them for you — there is nothing to type.
      </Note>
      {sources.length === 0 ? (
        <Card>
          <p className="py-2 text-sm text-stone-500">
            Add a portfolio and Trakt will work out what it expects to receive.
          </p>
        </Card>
      ) : (
        sources.map((source, index) => {
          const portfolio = draft.profile.portfolios.find(
            (p) => p.portfolio_id === source.portfolio_id,
          );
          return (
            <Card
              key={`${source.portfolio_id}-${source.dataset}-${index}`}
              title={`${portfolio?.display_name ?? source.portfolio_id} — ${label(source.dataset)}`}
            >
              <KeyValue label="Book" value={label(source.dataset)} />
              <KeyValue label="Expected every" value={label(source.frequency)} />
              <KeyValue label="Supplied by" value={source.source_system} />
              <KeyValue
                label="Included in regulatory reporting"
                value={source.regime_required ? "Yes" : "No"}
              />
              <KeyValue label="Status until the first delivery" value={label(source.status)} />
            </Card>
          );
        })
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Step 7 — Review
// --------------------------------------------------------------------------- //

function ReviewStep({
  review,
  reason,
  onReason,
  onApprove,
  approving,
}: {
  review: OnboardingReview;
  reason: string;
  onReason: (value: string) => void;
  onApprove: () => void;
  approving: boolean;
}) {
  return (
    <div className="space-y-6">
      <Note>{copy.onboarding.reviewIntro}</Note>

      {!review.ready && (
        <Note tone="warn">
          {review.problems.length} answer{review.problems.length === 1 ? "" : "s"} still needed
          before this can be saved.
        </Note>
      )}

      <Card title="What will be written">
        <ul className="space-y-4">
          {review.artefacts.map((artefact) => (
            <li key={artefact.rel} className="border-b border-stone-100 pb-4 last:border-0 last:pb-0">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-medium text-stone-900">{artefact.label}</span>
                <ActionChip action={artefact.action} />
              </div>
              <ul className="mt-2 space-y-1">
                {artefact.summary.map((line) => (
                  <li key={line} className="text-sm text-stone-500">
                    {line}
                  </li>
                ))}
              </ul>
              {artefact.records.length > 0 && (
                <ul className="mt-3 space-y-2">
                  {artefact.records.map((record) => (
                    <li
                      key={`${record.source_portfolio_id}-${record.dataset}-${record.frequency}`}
                      className="rounded-xl bg-stone-50 px-3 py-2 text-sm text-stone-600"
                    >
                      <span className="font-medium text-stone-800">
                        {record.portfolio_label || record.source_portfolio_id}
                      </span>{" "}
                      · {label(record.dataset)} · {label(record.frequency)}
                      {record.regime_required ? " · regulatory reporting" : ""}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      </Card>

      {review.changes.length > 0 && (
        <Card
          title={review.current_version > 0 ? "What changes" : "What will be recorded"}
          description={
            review.current_version > 0
              ? `Version ${review.current_version} stays exactly as it is. This becomes version ${review.next_version}.`
              : `This becomes version ${review.next_version}.`
          }
        >
          <ul className="space-y-2">
            {review.changes.slice(0, 40).map((change) => (
              <li
                key={change.path}
                className="flex flex-wrap items-baseline justify-between gap-2 border-b border-stone-100 py-2 last:border-0"
              >
                <span className="text-sm text-stone-600">{change.label}</span>
                <span className="text-sm">
                  {change.before && (
                    <span className="text-stone-400 line-through">{change.before}</span>
                  )}{" "}
                  <span className="font-medium text-stone-900">{change.after || "—"}</span>
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {review.unrepresented.length > 0 && (
        <Card
          title="Recorded, but not written into configuration"
          description="Trakt keeps these with the client's history. Nothing today reads them."
        >
          <ul className="space-y-2">
            {review.unrepresented.map((entry) => (
              <li key={`${entry.product}-${entry.key}`} className="text-sm text-stone-500">
                <span className="font-medium text-stone-700">{entry.label}</span> — {entry.reason}
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card title="Approve">
        <div className="space-y-4">
          <Field label={copy.onboarding.approveReason} required>
            <TextInput value={reason} onChange={onReason} placeholder="A short note for the record" />
          </Field>
          <PrimaryButton onClick={onApprove} disabled={!review.ready || !reason.trim() || approving}>
            {approving ? copy.onboarding.saving : copy.onboarding.approve}
          </PrimaryButton>
        </div>
      </Card>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// The wizard
// --------------------------------------------------------------------------- //

export function OnboardingWizardScreen() {
  const { id = "" } = useParams();
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [step, setStep] = useState<OnboardingStep>("client");
  const [reason, setReason] = useState("");
  const [approving, setApproving] = useState(false);
  const [review, setReview] = useState<OnboardingReview | null>(null);

  const load = useCallback(
    async () => ({
      draft: await client.getOnboardingDraft(id),
      reference: await client.getOnboardingReference(),
    }),
    [client, id],
  );
  const { data, error, loading, reload } = useLoad(load, [id]);
  const [draft, setDraft] = useState<OnboardingDraft | null>(null);
  const current = draft ?? data?.draft ?? null;
  const reference = data?.reference ?? null;

  const stepProblems = useMemo(
    () => (current ? (current.problems_by_step[step] ?? []) : []),
    [current, step],
  );

  const save = useCallback(
    async (nextStep: OnboardingStep, payload: Record<string, unknown>) => {
      if (!current) return;
      try {
        const updated = await client.saveOnboardingStep(
          id,
          nextStep,
          payload,
          current.client_id || undefined,
        );
        setDraft(updated);
      } catch (err) {
        toast.show(errorMessage(err), "error");
      }
    },
    [client, current, id, toast],
  );

  async function goTo(next: OnboardingStep) {
    setStep(next);
    if (next === "review" && current) {
      try {
        setReview(await client.reviewOnboardingDraft(id, current.client_id || undefined));
      } catch (err) {
        toast.show(errorMessage(err), "error");
      }
    }
  }

  async function approve() {
    if (!current) return;
    setApproving(true);
    try {
      const result = await client.approveOnboardingDraft(
        id,
        reason,
        current.client_id || undefined,
      );
      toast.show(copy.onboarding.approved, "success");
      navigate(`/onboarding/clients/${encodeURIComponent(result.client_id)}`);
    } catch (err) {
      toast.show(errorMessage(err), "error");
      setApproving(false);
    }
  }

  if (loading) return <Page title={copy.onboarding.wizardTitle}><Loading /></Page>;
  if (error || !current || !reference) {
    return (
      <Page title={copy.onboarding.wizardTitle}>
        <ErrorNote message={copy.onboarding.unavailable} onRetry={() => void reload()} />
      </Page>
    );
  }

  const steps = current.steps;
  const index = steps.findIndex((s) => s.key === step);
  const adopted = current.origin !== "new";

  return (
    <Page
      title={
        adopted
          ? current.profile.client.client_name || current.client_id
          : copy.onboarding.wizardTitle
      }
      subtitle={copy.onboarding.wizardSubtitle}
    >
      <StepRail steps={steps} current={step} onSelect={(s) => void goTo(s)} />

      {adopted && current.gaps.length > 0 && step === "client" && (
        <div className="mb-6">
          <Note tone="warn">
            Trakt filled in everything it already knew about this client. Still needed:{" "}
            {current.gaps.map((g) => g.label).join(", ")}.
          </Note>
        </div>
      )}

      {step === "client" && (
        <ClientStep
          profile={current.profile}
          problems={stepProblems}
          provenance={current.provenance}
          onChange={(patch) => void save("client", patch)}
        />
      )}
      {step === "portfolio" && (
        <PortfolioStep
          profile={current.profile}
          reference={reference}
          problems={stepProblems}
          onChange={(portfolios) => void save("portfolio", { portfolios })}
        />
      )}
      {step === "reporting" && (
        <ReportingStep
          draft={current}
          reference={reference}
          onChange={(products) => void save("reporting", { products })}
        />
      )}
      {step === "regime" && (
        <RegimeStep
          draft={current}
          reference={reference}
          onChange={(product, key, value) =>
            void save("regime", { regime: { [product]: { [key]: value } } })
          }
        />
      )}
      {step === "static_reporting" && (
        <StaticReportingStep
          profile={current.profile}
          provenance={current.provenance}
          onChange={(patch) => void save("static_reporting", patch)}
        />
      )}
      {step === "sources" && <SourcesStep draft={current} />}
      {step === "review" && review && (
        <ReviewStep
          review={review}
          reason={reason}
          onReason={setReason}
          onApprove={() => void approve()}
          approving={approving}
        />
      )}
      {step === "review" && !review && <Loading />}

      <div className="mt-8 flex items-center justify-between">
        <SecondaryButton
          disabled={index <= 0}
          onClick={() => void goTo(steps[Math.max(0, index - 1)].key)}
        >
          {copy.onboarding.back}
        </SecondaryButton>
        {step !== "review" && (
          <PrimaryButton onClick={() => void goTo(steps[Math.min(steps.length - 1, index + 1)].key)}>
            {copy.onboarding.next}
          </PrimaryButton>
        )}
      </div>
    </Page>
  );
}
