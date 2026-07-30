/**
 * Renders whatever the governed catalogue declares.
 *
 * There are no hard-coded field lists in this file. A field added to
 * `config/onboarding/field_catalogue.yaml` appears here — with its label, help,
 * type, options and conditional requirement — without a change to any component.
 * That is the point: the form is a projection of the information model, not a
 * second copy of it.
 */

import type {
  CaseAnswers,
  CaseProblem,
  CatalogueField,
  CatalogueSection,
} from "@/api/onboardingTypes";
import { isRequired } from "@/api/MockOnboarding";
import { Field, Select, TextInput, Toggle } from "./primitives";

export type FieldValue = string | boolean | string[];

/**
 * Which fields appear on the form.
 *
 * `inferred` is shown: Trakt fills it in and the operator can change it, which
 * is not the same as not asking. `derived` and `system_generated` are hidden —
 * they follow from another answer by a fixed relationship, and offering an
 * independent edit would invite two copies of one fact to disagree.
 */
export const SHOWN = ["client_supplied", "operator_supplied", "trakt_default", "inferred"];

/** True when this field is a question rather than something Trakt supplies. */
export const isAsked = (source: string) => SHOWN.includes(source);

function problemFor(
  problems: CaseProblem[],
  key: string,
  index: number | null = null,
): string | undefined {
  return problems.find((p) => p.field === key && (p.index ?? null) === index)?.message;
}

/** One catalogue field, rendered by its declared type. */
export function CatalogueFieldInput({
  field,
  value,
  problem,
  origin,
  required,
  onChange,
  entities,
  portfolios,
}: {
  field: CatalogueField;
  value: unknown;
  problem?: string;
  origin?: string;
  required: boolean;
  onChange: (value: FieldValue) => void;
  entities?: { value: string; label: string }[];
  portfolios?: { value: string; label: string }[];
}) {
  const text = value === undefined || value === null ? "" : String(value);
  const hint = [field.help, field.regime_code ? `Reported as ${field.regime_code}.` : ""]
    .filter(Boolean)
    .join(" ");

  if (field.type === "boolean") {
    return (
      <Toggle
        label={field.label}
        description={field.help}
        checked={value === true || text === "true"}
        onChange={onChange}
      />
    );
  }

  if (field.type === "multi_enum") {
    const chosen = Array.isArray(value) ? (value as string[]) : [];
    return (
      <Field label={field.label} required={required} hint={hint} problem={problem} origin={origin}>
        <div className="flex flex-wrap gap-2">
          {field.options.map((option) => {
            const on = chosen.includes(option.value);
            return (
              <button
                key={option.value}
                type="button"
                onClick={() =>
                  onChange(
                    on ? chosen.filter((v) => v !== option.value) : [...chosen, option.value],
                  )
                }
                className={
                  on
                    ? "rounded-full border border-stone-900 bg-stone-900 px-3 py-1 text-xs font-medium text-white"
                    : "rounded-full border border-stone-300 px-3 py-1 text-xs font-medium text-stone-600 hover:border-stone-500"
                }
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </Field>
    );
  }

  if (field.type === "entity_reference" || field.type === "portfolio_reference") {
    const options = field.type === "entity_reference" ? (entities ?? []) : (portfolios ?? []);
    return (
      <Field label={field.label} required={required} hint={hint} problem={problem} origin={origin}>
        <Select value={text} onChange={onChange} options={options} placeholder="Choose" />
      </Field>
    );
  }

  if (field.type === "enum" || field.options.length > 0) {
    return (
      <Field label={field.label} required={required} hint={hint} problem={problem} origin={origin}>
        <Select
          value={text}
          onChange={onChange}
          options={field.options}
          placeholder="Choose a value"
          invalid={Boolean(problem)}
        />
      </Field>
    );
  }

  if (field.type === "yes_no") {
    return (
      <Field label={field.label} required={required} hint={hint} problem={problem} origin={origin}>
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

  if (field.type === "multiline") {
    return (
      <Field label={field.label} required={required} hint={hint} problem={problem} origin={origin}>
        <textarea
          className="w-full rounded-xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-900 outline-none transition-colors focus:border-stone-900"
          rows={3}
          value={text}
          onChange={(e) => onChange(e.target.value)}
        />
      </Field>
    );
  }

  return (
    <Field label={field.label} required={required} hint={hint} problem={problem} origin={origin}>
      <TextInput value={text} onChange={onChange} invalid={Boolean(problem)} />
    </Field>
  );
}

/** A non-repeating section: one block of answers. */
export function SectionForm({
  section,
  answers,
  problems,
  provenance,
  onChange,
  entities,
  portfolios,
}: {
  section: CatalogueSection;
  answers: CaseAnswers;
  problems: CaseProblem[];
  provenance: Record<string, string>;
  onChange: (patch: Record<string, FieldValue>) => void;
  entities?: { value: string; label: string }[];
  portfolios?: { value: string; label: string }[];
}) {
  const block = (answers[section.key] ?? {}) as Record<string, unknown>;
  const asked = section.fields.filter((f) => SHOWN.includes(f.source));
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {asked.map((field) => (
        <CatalogueFieldInput
          key={field.key}
          field={field}
          value={block[field.key]}
          required={isRequired(field, answers, block)}
          problem={problemFor(problems, field.key)}
          origin={provenance[`${section.key}.${field.key}`]}
          entities={entities}
          portfolios={portfolios}
          onChange={(value) => onChange({ [field.key]: value })}
        />
      ))}
    </div>
  );
}

/** One item of a repeatable section. */
export function ItemForm({
  section,
  item,
  index,
  answers,
  problems,
  onChange,
  entities,
  portfolios,
}: {
  section: CatalogueSection;
  item: Record<string, unknown>;
  index: number;
  answers: CaseAnswers;
  problems: CaseProblem[];
  onChange: (patch: Record<string, FieldValue>) => void;
  entities?: { value: string; label: string }[];
  portfolios?: { value: string; label: string }[];
}) {
  const asked = section.fields.filter((f) => SHOWN.includes(f.source));
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {asked.map((field) => (
        <CatalogueFieldInput
          key={field.key}
          field={field}
          value={item[field.key]}
          required={isRequired(field, answers, item)}
          problem={problemFor(problems, field.key, index)}
          entities={entities}
          portfolios={portfolios}
          onChange={(value) => onChange({ [field.key]: value })}
        />
      ))}
    </div>
  );
}
