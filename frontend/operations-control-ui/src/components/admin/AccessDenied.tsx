import { ShieldOff } from "lucide-react";
import { Page } from "@/components/Page";
import { copy } from "@/lib/copy";

/** Plain-English refusal shown when a non-administrator reaches an admin route
 *  directly, or when the backend answers 403. */
export function AccessDenied() {
  return (
    <Page title={copy.admin.accessDeniedTitle}>
      <div
        role="alert"
        className="flex items-start gap-3 rounded-2xl border border-stone-200 bg-white px-6 py-8"
      >
        <ShieldOff className="mt-0.5 h-5 w-5 shrink-0 text-stone-400" aria-hidden />
        <div>
          <p className="text-base text-stone-800">{copy.admin.accessDenied}</p>
          <p className="mt-1 text-sm text-stone-500">{copy.admin.accessDeniedHelp}</p>
        </div>
      </div>
    </Page>
  );
}
