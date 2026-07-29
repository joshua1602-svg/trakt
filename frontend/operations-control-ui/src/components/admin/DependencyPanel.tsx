import { Link2 } from "lucide-react";
import type { Dependency } from "@/api/adminTypes";
import { copy } from "@/lib/copy";
import { EmptyNote, SectionHeading } from "./primitives";

/** What a package relies on, and what relies on it. */
export function DependencyPanel({ dependencies }: { dependencies: Dependency[] }) {
  return (
    <section>
      <SectionHeading>{copy.admin.sections.dependencies}</SectionHeading>
      {dependencies.length === 0 ? (
        <EmptyNote>{copy.admin.compatibility.none}</EmptyNote>
      ) : (
        <ul className="space-y-2">
          {dependencies.map((dependency) => (
            <li
              key={dependency.name}
              className="flex items-start gap-3 rounded-xl border border-stone-200 bg-white px-4 py-3"
            >
              <Link2 className="mt-0.5 h-4 w-4 shrink-0 text-stone-400" aria-hidden />
              <div>
                <p className="text-sm font-medium text-stone-900">{dependency.name}</p>
                <p className="mt-0.5 text-sm text-stone-500">{dependency.relationship}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
