import { TraktWordmark } from "@/components/site/TraktWordmark";

export function Footer() {
  return (
    <footer className="border-t border-line px-5 py-10 sm:px-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-center gap-2.5">
          <TraktWordmark size={22} />
        </div>
        <div className="max-w-lg space-y-2 text-xs leading-relaxed text-ink-500">
          <p>
            Portfolio intelligence for specialist lenders, non-bank lenders,
            private-credit managers, servicing businesses and securitisation
            participants.
          </p>
          <p>
            The demonstration on this page uses a wholly synthetic portfolio. No
            client or consumer information is displayed, and no data is accepted from
            visitors.
          </p>
          <p>© {new Date().getFullYear()} Trakt. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
