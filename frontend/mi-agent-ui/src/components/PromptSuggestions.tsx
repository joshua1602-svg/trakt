import { ArrowUpRight } from "lucide-react";
import { PROMPT_SUGGESTIONS } from "@/data/agentEngine";

export function PromptSuggestions({
  onPick,
}: {
  onPick: (prompt: string) => void;
}) {
  return (
    <div>
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-ink-500">
        Suggested questions
      </div>
      <div className="flex flex-col gap-1.5">
        {PROMPT_SUGGESTIONS.map((s) => (
          <button
            key={s.label}
            type="button"
            onClick={() => onPick(s.label)}
            className="group flex items-center gap-2 rounded-lg border border-[var(--color-line)] bg-navy-900/50 px-3 py-2 text-left text-[12px] text-ink-300 transition-colors hover:border-peri-400/40 hover:bg-navy-800/60 hover:text-ink-100"
          >
            <span className="flex-1">{s.label}</span>
            <ArrowUpRight
              size={14}
              className="shrink-0 text-ink-500 transition-colors group-hover:text-peri-300"
            />
          </button>
        ))}
      </div>
    </div>
  );
}
