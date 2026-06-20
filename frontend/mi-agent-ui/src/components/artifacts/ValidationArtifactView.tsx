import { AlertOctagon, AlertTriangle, CheckCircle2, Info } from "lucide-react";
import type { ValidationArtifact, ValidationSeverity } from "@/domain";
import { Badge } from "@/components/ui";
import { cn } from "@/lib/utils";

const SEV: Record<
  ValidationSeverity,
  { icon: typeof Info; tone: "rose" | "amber" | "navy" | "mint"; label: string; color: string }
> = {
  blocker: { icon: AlertOctagon, tone: "rose", label: "Blocker", color: "text-rose-400" },
  warning: { icon: AlertTriangle, tone: "amber", label: "Warning", color: "text-amber-400" },
  info: { icon: Info, tone: "navy", label: "Info", color: "text-peri-300" },
  pass: { icon: CheckCircle2, tone: "mint", label: "Pass", color: "text-mint-400" },
};

function Stat({ value, label, color }: { value: number | string; label: string; color: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-850/60 px-3 py-2.5 text-center">
      <div className={cn("font-mono text-xl font-semibold tabular-nums", color)}>{value}</div>
      <div className="mt-0.5 text-[10px] font-medium uppercase tracking-wider text-ink-400">{label}</div>
    </div>
  );
}

export function ValidationArtifactView({ artifact }: { artifact: ValidationArtifact }) {
  const order: ValidationSeverity[] = ["blocker", "warning", "info", "pass"];
  const issues = [...artifact.issues].sort((a, b) => order.indexOf(a.severity) - order.indexOf(b.severity));

  return (
    <div>
      <div className="grid grid-cols-4 gap-2.5">
        <Stat value={artifact.summary.blockers} label="Blockers" color="text-rose-400" />
        <Stat value={artifact.summary.warnings} label="Warnings" color="text-amber-400" />
        <Stat value={artifact.summary.passed} label="Passed" color="text-mint-400" />
        <Stat value={`${artifact.summary.coverage}%`} label="Coverage" color="text-peri-300" />
      </div>

      <ul className="mt-3 flex flex-col gap-2">
        {issues.map((issue) => {
          const sev = SEV[issue.severity];
          const Icon = sev.icon;
          return (
            <li key={issue.id} className="rounded-lg border border-[var(--color-line-soft)] bg-navy-850/40 p-3">
              <div className="flex items-start gap-2.5">
                <Icon size={16} className={cn("mt-0.5 shrink-0", sev.color)} />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-ink-100">{issue.title}</span>
                    <Badge tone={sev.tone}>{sev.label}</Badge>
                    {issue.affected != null && <span className="text-[11px] text-ink-500">{issue.affected} affected</span>}
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-ink-400">{issue.detail}</p>
                  <div className="mt-1.5 flex items-center gap-2 font-mono text-[10px] text-ink-500">
                    <span className="rounded bg-navy-800 px-1.5 py-0.5">{issue.code}</span>
                    <span>{issue.scope}</span>
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
