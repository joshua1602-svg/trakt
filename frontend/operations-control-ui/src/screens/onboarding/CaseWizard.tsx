import { useCallback, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Check, Plus, Send, Trash2 } from "lucide-react";
import clsx from "clsx";
import { useOpsClient } from "@/api/context";
import type {
  CasePreview,
  CatalogueSection,
  ChecklistRow,
  OnboardingCase,
  OnboardingReference,
} from "@/api/onboardingTypes";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import {
  CatalogueFieldInput,
  ItemForm,
  SectionForm,
  isAsked,
  type FieldValue,
} from "@/components/onboarding/CatalogueForm";
import {
  ActionChip,
  Card,
  Field,
  KeyValue,
  Note,
  PrimaryButton,
  SecondaryButton,
  SectionHeading,
  TextInput,
  Toggle,
} from "@/components/onboarding/primitives";
import { WithdrawDialog } from "@/components/onboarding/WithdrawDialog";
import { Page } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { copy } from "@/lib/copy";
import { errorMessage, useLoad } from "@/lib/useLoad";

const label = (v: string) => v.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

function StepRail({
  steps,
  current,
  onSelect,
}: {
  steps: OnboardingCase["steps"];
  current: string;
  onSelect: (step: string) => void;
}) {
  return (
    <ol className="mb-6 -mx-1 flex snap-x gap-2 overflow-x-auto px-1 pb-2">
      {steps.map((step, index) => {
        const active = step.key === current;
        return (
          <li key={step.key} className="snap-start">
            <button
              type="button"
              onClick={() => onSelect(step.key)}
              className={clsx(
                "flex shrink-0 items-center gap-2 rounded-xl border px-3 py-2 text-sm transition-colors",
                active
                  ? "border-stone-900 bg-stone-900 text-white"
                  : "border-stone-200 bg-white text-stone-600 hover:border-stone-400",
              )}
            >
              <span
                className={clsx(
                  "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                  step.problems > 0
                    ? "bg-amber-100 text-amber-800"
                    : active
                      ? "bg-white text-stone-900"
                      : "bg-stone-100 text-stone-500",
                )}
              >
                {step.problems === 0 ? <Check className="h-3 w-3" aria-hidden /> : index + 1}
              </span>
              <span className="whitespace-nowrap">{step.label}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function RepeatableStep({
  section,
  current,
  onSave,
  entities,
  portfolios,
}: {
  section: CatalogueSection;
  current: OnboardingCase;
  onSave: (items: Record<string, unknown>[]) => void;
  entities: { value: string; label: string }[];
  portfolios: { value: string; label: string }[];
}) {
  const items = (current.answers[section.key] ?? []) as Record<string, unknown>[];
  const problems = current.by_step[section.key] ?? [];

  function update(index: number, patch: Record<string, FieldValue>) {
    onSave(items.map((item, i) => (i === index ? { ...item, ...patch } : item)));
  }

  return (
    <div className="space-y-6">
      {section.help && <Note>{section.help}</Note>}
      {items.map((item, index) => (
        <Card
          key={String(item[section.repeatable_key] ?? index)}
          title={String(item[section.item_label_field] ?? "") || `${section.label} ${index + 1}`}
          actions={
            <button
              type="button"
              onClick={() => onSave(items.filter((_, i) => i !== index))}
              className="flex items-center gap-1 text-sm text-stone-500 hover:text-red-600"
            >
              <Trash2 className="h-4 w-4" aria-hidden />
              Remove
            </button>
          }
        >
          <ItemForm
            section={section}
            item={item}
            index={index}
            answers={current.answers}
            problems={problems}
            entities={entities}
            portfolios={portfolios}
            onChange={(patch) => update(index, patch)}
          />
        </Card>
      ))}
      <SecondaryButton onClick={() => onSave([...items, {}])}>
        <span className="flex items-center gap-2">
          <Plus className="h-4 w-4" aria-hidden />
          Add
        </span>
      </SecondaryButton>
    </div>
  );
}

function ReportingStep({
  current,
  reference,
  onSave,
}: {
  current: OnboardingCase;
  reference: OnboardingReference;
  onSave: (products: string[]) => void;
}) {
  const chosen = ((current.answers.reporting ?? {}) as { products?: string[] }).products ?? [];
  return (
    <div className="space-y-6">
      <Note>{copy.onboarding.reportingIntro}</Note>
      <Card title={copy.onboarding.reportingHeading}>
        <div className="space-y-3">
          {Object.entries(reference.catalogue.regime_products).map(([key, product]) => {
            const state = current.product_eligibility[key];
            const alwaysOn = Boolean(product.always_applies);
            return (
              <Toggle
                key={key}
                label={product.label}
                description={
                  alwaysOn ? `Always prepared. ${product.derived_from ?? ""}` : state?.reason
                }
                checked={alwaysOn || chosen.includes(key)}
                disabled={alwaysOn || !(state?.eligible ?? true)}
                onChange={(on) =>
                  onSave(on ? [...chosen, key] : chosen.filter((k) => k !== key))
                }
              />
            );
          })}
        </div>
      </Card>
    </div>
  );
}

function RegimeStep({
  current,
  reference,
  onSave,
}: {
  current: OnboardingCase;
  reference: OnboardingReference;
  onSave: (product: string, key: string, value: FieldValue) => void;
}) {
  const section = reference.catalogue.sections.find((s) => s.from_regime);
  const chosen = ((current.answers.reporting ?? {}) as { products?: string[] }).products ?? [];
  const held = (current.answers.regime ?? {}) as Record<string, Record<string, unknown>>;
  // Only products with something left to ASK. A regime whose every standing
  // value follows from an answer already given needs no step of its own.
  const withFields = chosen.filter((key) =>
    (section?.fields ?? []).some((f) => f.product === key && isAsked(f.source)),
  );

  if (!section || withFields.length === 0) {
    return <Note>{copy.onboarding.noRegimeFields}</Note>;
  }

  return (
    <div className="space-y-6">
      <Note>{section.help}</Note>
      {withFields.map((product) => {
        const fields = section.fields.filter(
          (f) => f.product === product && isAsked(f.source),
        );
        const productSpec = reference.catalogue.regime_products[product];
        return (
          <Card key={product} title={productSpec?.label ?? label(product)}>
            <div className="grid gap-4 sm:grid-cols-2">
              {fields.map((field) => {
                const problem = current.by_step.regime?.find(
                  (p) => p.field === `${product}.${field.key}`,
                )?.message;
                return (
                  <div key={field.key} className={field.type === "boolean" ? "sm:col-span-2" : ""}>
                    <CatalogueFieldInput
                      field={field}
                      value={(held[product] ?? {})[field.key]}
                      problem={problem}
                      required={field.required}
                      origin={current.provenance[`regime.${product}.${field.key}`]}
                      onChange={(value) => onSave(product, field.key, value)}
                    />
                  </div>
                );
              })}
            </div>
            {(productSpec?.unrepresented ?? []).length > 0 && (
              <div className="mt-6 border-t border-stone-100 pt-4">
                <SectionHeading>{copy.onboarding.notHeldHere}</SectionHeading>
                <ul className="space-y-2">
                  {(productSpec?.unrepresented ?? []).map((entry) => (
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

function SourcesStep({
  section,
  current,
  onSave,
  onAddPipeline,
}: {
  section: CatalogueSection;
  current: OnboardingCase;
  onSave: (items: Record<string, unknown>[]) => void;
  onAddPipeline: (portfolioId: string) => void;
}) {
  const items = (current.answers.sources ?? []) as Record<string, unknown>[];
  const portfolios = (current.answers.portfolios ?? []) as Record<string, unknown>[];
  const problems = current.by_step.sources ?? [];

  return (
    <div className="space-y-6">
      <Note>{copy.onboarding.sourcesIntro}</Note>
      {items.length === 0 && (
        <Card>
          <p className="py-2 text-sm text-stone-500">{copy.onboarding.noSources}</p>
        </Card>
      )}
      {items.map((item, index) => {
        const portfolio = portfolios.find((p) => p.portfolio_id === item.portfolio_id);
        return (
          <Card
            key={String(item.source_key ?? index)}
            title={`${String(portfolio?.display_name ?? item.portfolio_id)} — ${label(String(item.dataset))}`}
          >
            <ItemForm
              section={section}
              item={item}
              index={index}
              answers={current.answers}
              problems={problems}
              onChange={(patch) =>
                onSave(items.map((s, i) => (i === index ? { ...s, ...patch } : s)))
              }
            />
            {Boolean(item.expected_location) && (
              <div className="mt-4 border-t border-stone-100 pt-3">
                <KeyValue
                  label={copy.onboarding.expectedLocation}
                  value={
                    <span className="break-all font-mono text-xs">
                      {String(item.expected_location)}
                    </span>
                  }
                />
              </div>
            )}
          </Card>
        );
      })}
      {portfolios
        .filter((p) => !items.some((s) => s.portfolio_id === p.portfolio_id && s.dataset === "pipeline"))
        .map((p) => (
          <SecondaryButton
            key={String(p.portfolio_id)}
            onClick={() => onAddPipeline(String(p.portfolio_id))}
          >
            <span className="flex items-center gap-2">
              <Plus className="h-4 w-4" aria-hidden />
              {copy.onboarding.addPipeline} {String(p.display_name ?? p.portfolio_id)}
            </span>
          </SecondaryButton>
        ))}
    </div>
  );
}

function ChecklistPanel({
  current,
  onRequest,
  busy,
}: {
  current: OnboardingCase;
  onRequest: (items: ChecklistRow[]) => void;
  busy: boolean;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const rows = current.client_checklist;
  const key = (r: ChecklistRow) => `${r.section}/${r.field}/${r.index}`;

  return (
    <Card
      title={copy.onboarding.checklistHeading}
      description={copy.onboarding.checklistDescription}
    >
      {rows.length === 0 ? (
        <p className="py-2 text-sm text-stone-500">{copy.onboarding.checklistEmpty}</p>
      ) : (
        <>
          <ul className="mb-4 space-y-2">
            {rows.map((row) => (
              <li key={key(row)}>
                <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-stone-200 px-3 py-2 hover:border-stone-400">
                  <input
                    type="checkbox"
                    className="mt-1 h-4 w-4 shrink-0 accent-stone-900"
                    checked={selected.has(key(row))}
                    onChange={(e) => {
                      const next = new Set(selected);
                      if (e.target.checked) next.add(key(row));
                      else next.delete(key(row));
                      setSelected(next);
                    }}
                  />
                  <span className="min-w-0">
                    <span className="block break-words text-sm font-medium text-stone-900">
                      {row.label}
                    </span>
                    <span className="block text-xs text-stone-500">
                      {row.section_label}
                      {row.evidence_required && " · evidence required"}
                    </span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <PrimaryButton
            disabled={selected.size === 0 || busy}
            onClick={() => onRequest(rows.filter((r) => selected.has(key(r))))}
          >
            <span className="flex items-center gap-2">
              <Send className="h-4 w-4" aria-hidden />
              {copy.onboarding.requestSelected}
            </span>
          </PrimaryButton>
        </>
      )}

      {current.information_requests.length > 0 && (
        <div className="mt-6 border-t border-stone-100 pt-4">
          <SectionHeading>{copy.onboarding.requestsHeading}</SectionHeading>
          <ul className="space-y-2">
            {current.information_requests.map((request) => (
              <li key={request.request_id} className="text-sm text-stone-600">
                <span className="font-medium text-stone-800">
                  {request.items.length} item{request.items.length === 1 ? "" : "s"}
                </span>{" "}
                · {label(request.status)} · requested by {request.requested_by}
                {request.evidence.length > 0 && ` · ${request.evidence.length} document(s)`}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}

function ReviewStep({
  preview,
  reason,
  onReason,
  onApprove,
  onActivate,
  busy,
  status,
}: {
  preview: CasePreview;
  reason: string;
  onReason: (v: string) => void;
  onApprove: () => void;
  onActivate: () => void;
  busy: boolean;
  status: string;
}) {
  const approved = status === "approved" || status === "activated";
  return (
    <div className="space-y-6">
      <Note>{copy.onboarding.reviewIntro}</Note>

      {!preview.ready && (
        <Note tone="warn">
          {preview.blocking.length > 0
            ? `${preview.blocking.length} answer${preview.blocking.length === 1 ? "" : "s"} still needed before this can be approved.`
            : copy.onboarding.stillOutstanding}
        </Note>
      )}

      <Card title={copy.onboarding.willBeCreated}>
        <ul className="space-y-4">
          {preview.artefacts.map((artefact) => (
            <li key={artefact.rel} className="border-b border-stone-100 pb-4 last:border-0 last:pb-0">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-medium text-stone-900">{artefact.label}</span>
                <ActionChip action={artefact.action} />
              </div>
              <ul className="mt-2 space-y-1">
                {artefact.summary.map((line) => (
                  <li key={line} className="break-words text-sm text-stone-500">
                    {line}
                  </li>
                ))}
              </ul>
              {artefact.records.length > 0 && (
                <ul className="mt-3 space-y-2">
                  {artefact.records.map((record) => (
                    <li
                      key={`${record.source_portfolio_id}-${record.dataset}`}
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

      {preview.generated_identifiers.length > 0 && (
        <Card
          title={copy.onboarding.generatedHeading}
          description={copy.onboarding.generatedDescription}
        >
          {preview.generated_identifiers.map((row) => (
            <KeyValue
              key={`${row.label}-${row.value}`}
              label={row.label}
              value={<span className="break-all font-mono text-xs">{row.value}</span>}
            />
          ))}
        </Card>
      )}

      {preview.defaults_used.length > 0 && (
        <Card title={copy.onboarding.defaultsHeading}>
          {preview.defaults_used.map((row) => (
            <KeyValue key={row.label} label={row.label} value={row.value} />
          ))}
        </Card>
      )}

      {preview.changes.length > 0 && (
        <Card
          title={preview.current_version > 0 ? copy.onboarding.whatChanges : copy.onboarding.whatRecorded}
          description={
            preview.current_version > 0
              ? `Version ${preview.current_version} stays exactly as it is. This becomes version ${preview.next_version}.`
              : `This becomes version ${preview.next_version}.`
          }
        >
          <ul className="space-y-2">
            {preview.changes.slice(0, 40).map((change) => (
              <li
                key={change.path}
                className="flex flex-wrap items-baseline justify-between gap-2 border-b border-stone-100 py-2 last:border-0"
              >
                <span className="break-words text-sm text-stone-600">{change.label}</span>
                <span className="break-words text-sm">
                  {change.before && <span className="text-stone-400 line-through">{change.before}</span>}{" "}
                  <span className="font-medium text-stone-900">{change.after || "—"}</span>
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {preview.unrepresented.length > 0 && (
        <Card
          title={copy.onboarding.unrepresentedHeading}
          description={copy.onboarding.unrepresentedDescription}
        >
          <ul className="space-y-2">
            {preview.unrepresented.map((entry, i) => (
              <li key={`${entry.label}-${i}`} className="break-words text-sm text-stone-500">
                <span className="font-medium text-stone-700">{entry.label}</span> — {entry.reason}
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card title={approved ? copy.onboarding.activateHeading : copy.onboarding.approveHeading}>
        <div className="space-y-4">
          {!approved && (
            <Field label={copy.onboarding.approveReason} required>
              <TextInput
                value={reason}
                onChange={onReason}
                placeholder={copy.onboarding.reasonPlaceholder}
              />
            </Field>
          )}
          {approved ? (
            <>
              <Note>{copy.onboarding.approvedNote}</Note>
              <PrimaryButton onClick={onActivate} disabled={busy || status === "activated"}>
                {copy.onboarding.activate}
              </PrimaryButton>
            </>
          ) : (
            <PrimaryButton onClick={onApprove} disabled={!preview.ready || !reason.trim() || busy}>
              {copy.onboarding.approve}
            </PrimaryButton>
          )}
        </div>
      </Card>
    </div>
  );
}

/** The governed onboarding case: one wizard for new clients, migrations and
 *  amendments, because all three gather the same information. */
export function OnboardingCaseScreen() {
  const { id = "" } = useParams();
  const client = useOpsClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [step, setStep] = useState("client");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<CasePreview | null>(null);
  const [live, setLive] = useState<OnboardingCase | null>(null);
  const [confirmWithdraw, setConfirmWithdraw] = useState(false);

  const load = useCallback(
    async () => ({
      onboardingCase: await client.getCase(id),
      reference: await client.getOnboardingReference(),
    }),
    [client, id],
  );
  const { data, error, loading, reload } = useLoad(load, [id]);
  const current = live ?? data?.onboardingCase ?? null;
  const reference = data?.reference ?? null;

  const entityOptions = useMemo(
    () =>
      (((current?.answers.entities ?? []) as Record<string, unknown>[]) ?? []).map((e) => ({
        value: String(e.entity_id ?? ""),
        label: String(e.legal_name ?? "") || "Unnamed entity",
      })),
    [current],
  );
  const portfolioOptions = useMemo(
    () =>
      (((current?.answers.portfolios ?? []) as Record<string, unknown>[]) ?? []).map((p) => ({
        value: String(p.portfolio_id ?? ""),
        label: String(p.display_name ?? p.portfolio_id ?? ""),
      })),
    [current],
  );

  const save = useCallback(
    async (stepKey: string, payload: Record<string, unknown>) => {
      try {
        setLive(await client.saveCaseStep(id, stepKey, payload));
      } catch (err) {
        toast.show(errorMessage(err), "error");
      }
    },
    [client, id, toast],
  );

  async function goTo(next: string) {
    setStep(next);
    if (next === "review") {
      try {
        setPreview(await client.getCasePreview(id));
      } catch (err) {
        toast.show(errorMessage(err), "error");
      }
    }
  }

  async function act(fn: () => Promise<unknown>, after?: () => void) {
    setBusy(true);
    try {
      await fn();
      setLive(await client.getCase(id));
      setPreview(await client.getCasePreview(id));
      after?.();
    } catch (err) {
      toast.show(errorMessage(err), "error");
    } finally {
      setBusy(false);
    }
  }

  if (loading) {
    return (
      <Page title={copy.onboarding.caseTitle}>
        <Loading />
      </Page>
    );
  }
  if (error || !current || !reference) {
    return (
      <Page title={copy.onboarding.caseTitle}>
        <ErrorNote message={copy.onboarding.unavailable} onRetry={() => void reload()} />
      </Page>
    );
  }

  const section = reference.catalogue.sections.find((s) => s.key === step);
  const index = current.steps.findIndex((s) => s.key === step);
  const named = current.client_name && current.client_name !== "Not yet named";
  // The two statuses the case model itself treats as terminal.
  const finished = current.status === "activated" || current.status === "withdrawn";

  return (
    <Page
      title={named ? current.client_name : copy.onboarding.caseTitle}
      subtitle={`${current.case_id} · ${current.kind_label} · ${current.status_label}`}
    >
      <StepRail steps={current.steps} current={step} onSelect={(s) => void goTo(s)} />

      {current.kind === "migration" && step === "client" && (
        <div className="mb-6">
          <Note tone="warn">{copy.onboarding.migrationNote}</Note>
        </div>
      )}

      {step === "review" ? (
        preview ? (
          <ReviewStep
            preview={preview}
            reason={reason}
            onReason={setReason}
            busy={busy}
            status={current.status}
            onApprove={() =>
              void act(() => client.approveCase(id, reason), () =>
                toast.show(copy.onboarding.approvedToast, "success"),
              )
            }
            onActivate={() =>
              void act(
                async () => {
                  const result = await client.activateCase(id);
                  navigate(`/onboarding/clients/${encodeURIComponent(result.client_id)}`);
                },
                () => toast.show(copy.onboarding.activatedToast, "success"),
              )
            }
          />
        ) : (
          <Loading />
        )
      ) : step === "reporting" ? (
        <ReportingStep
          current={current}
          reference={reference}
          onSave={(products) => void save("reporting", { products })}
        />
      ) : step === "regime" ? (
        <RegimeStep
          current={current}
          reference={reference}
          onSave={(product, key, value) =>
            void save("regime", { regime: { [product]: { [key]: value } } })
          }
        />
      ) : step === "sources" && section ? (
        <SourcesStep
          section={section}
          current={current}
          onSave={(items) => void save("sources", { sources: items })}
          onAddPipeline={(pid) => void act(() => client.addPipelineBook(id, pid))}
        />
      ) : section?.repeatable ? (
        <RepeatableStep
          section={section}
          current={current}
          entities={entityOptions}
          portfolios={portfolioOptions}
          onSave={(items) => void save(section.key, { [section.key]: items })}
        />
      ) : section ? (
        <div className="space-y-6">
          {section.help && <Note>{section.help}</Note>}
          <Card title={section.label}>
            <SectionForm
              section={section}
              answers={current.answers}
              problems={current.by_step[section.key] ?? []}
              provenance={current.provenance}
              entities={entityOptions}
              portfolios={portfolioOptions}
              onChange={(patch) => void save(section.key, patch)}
            />
          </Card>
          {section.key === "contacts" && (
            <ChecklistPanel
              current={current}
              busy={busy}
              onRequest={(items) => void act(() => client.createInformationRequest(id, items))}
            />
          )}
        </div>
      ) : null}

      <div className="mt-8 flex flex-wrap items-center justify-between gap-3">
        <SecondaryButton
          disabled={index <= 0}
          onClick={() => void goTo(current.steps[Math.max(0, index - 1)].key)}
        >
          {copy.onboarding.back}
        </SecondaryButton>
        {step !== "review" && (
          <PrimaryButton
            onClick={() =>
              void goTo(current.steps[Math.min(current.steps.length - 1, index + 1)].key)
            }
          >
            {copy.onboarding.next}
          </PrimaryButton>
        )}
      </div>

      {/* Available at every step, not only at the end: the moment an operator
          decides to abandon a case is exactly the moment they should not have
          to walk through the rest of the wizard to say so. */}
      {current.status === "withdrawn" && (
        <div className="mt-8">
          <Note tone="warn">{copy.onboarding.withdrawnNote}</Note>
        </div>
      )}
      {!finished && (
        <div className="mt-8 border-t border-stone-200 pt-6">
          <button
            type="button"
            onClick={() => setConfirmWithdraw(true)}
            className="text-sm font-medium text-stone-500 underline-offset-4 hover:text-stone-900 hover:underline"
          >
            {copy.onboarding.withdraw}
          </button>
        </div>
      )}

      {confirmWithdraw && (
        <WithdrawDialog
          kind={current.kind}
          busy={busy}
          onCancel={() => setConfirmWithdraw(false)}
          onConfirm={(why) =>
            void act(
              () => client.withdrawCase(id, why),
              () => {
                setConfirmWithdraw(false);
                toast.show(copy.onboarding.withdrawnToast, "success");
                navigate("/onboarding");
              },
            )
          }
        />
      )}
    </Page>
  );
}
