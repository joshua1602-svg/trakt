import { useState, type FormEvent } from "react";
import { KeyRound } from "lucide-react";
import { copy } from "@/lib/copy";

/** Minimal full-screen sign-in card: paste an access key, nothing else. */
export function SignIn({ onSubmit }: { onSubmit: (token: string) => void }) {
  const [value, setValue] = useState("");

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const token = value.trim();
    if (token) onSubmit(token);
  }

  return (
    <div className="flex min-h-full items-center justify-center bg-stone-50 px-6">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm rounded-2xl border border-stone-200 bg-white p-8 shadow-sm"
      >
        <div className="mb-6 flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-stone-900 text-white">
            <KeyRound className="h-5 w-5" aria-hidden />
          </span>
          <h1 className="text-lg font-semibold text-stone-900">{copy.signIn.title}</h1>
        </div>
        <label className="mb-1 block text-sm font-medium text-stone-700" htmlFor="access-key">
          {copy.signIn.prompt}
        </label>
        <p className="mb-3 text-xs text-stone-500">{copy.signIn.helper}</p>
        <input
          id="access-key"
          type="password"
          autoComplete="off"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder={copy.signIn.placeholder}
          className="mb-4 w-full rounded-xl border border-stone-300 px-3 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
        />
        <button
          type="submit"
          disabled={!value.trim()}
          className="w-full rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {copy.signIn.button}
        </button>
      </form>
    </div>
  );
}
