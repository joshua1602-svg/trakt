import { TraktWordmark } from "@/components/site/TraktWordmark";

export function Footer() {
  return (
    <footer className="border-t border-line px-5 py-10 sm:px-8 lg:px-12">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-center gap-2.5">
          <TraktWordmark size={22} />
        </div>
        <div className="max-w-lg space-y-2 text-small leading-relaxed text-ink-500">
          {/* Aligned with the strapline's first half so the site runs one
              product description, not two. The audience list is the part the
              strapline does not carry, which is why it stays. */}
          <p>
            Agentic portfolio intelligence for specialist lenders, non-bank
            lenders, private-credit managers, servicing businesses and
            securitisation participants.
          </p>
          <p>© {new Date().getFullYear()} Trakt. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
