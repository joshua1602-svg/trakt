/**
 * The Trakt wordmark.
 *
 * The mark is the existing brand asset from
 * `frontend/mi-agent-ui/public/trakt-mark.svg`, inlined so it needs no network
 * request and cannot shift layout. The typeface is the product UI's stack.
 */
export function TraktWordmark({ size = 26 }: { size?: number }) {
  return (
    <>
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        fill="none"
        role="img"
        aria-label="Trakt"
        focusable="false"
      >
        <rect width="32" height="32" rx="7" fill="#232D55" />
        <path
          d="M8 11h16M16 11v13"
          stroke="#919DD1"
          strokeWidth="2.6"
          strokeLinecap="round"
        />
        <circle cx="16" cy="24" r="1.8" fill="#fff" />
      </svg>
      <span className="text-[17px] font-semibold tracking-tight text-ink-100">trakt</span>
    </>
  );
}
