/**
 * identity — deriving the signed-in user's display from the API principal.
 *
 * The MI API `/me` endpoint echoes the Entra-verified principal (see
 * `mi_agent_api/auth.py`): `{ authenticated, user, roles, isOperator }`, where
 * `user` is the best available name/email claim. The UI derives a compact
 * display name and the role-based control visibility from that — never a
 * hardcoded label.
 */

export interface UserIdentity {
  authenticated: boolean;
  /** Best available name or email claim from the principal. */
  user?: string | null;
  roles?: string[];
  isOperator?: boolean;
}

/** True when the string looks like an email address (has an @ with text either side). */
function isEmail(value: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.trim());
}

/**
 * Compact display name: "Joshua Hall" → "J. Hall", "Jane Smith" → "J. Smith".
 * A single token is returned as-is. When only an email claim is available it is
 * the fallback (per spec: "Fallback to email if display name claim is
 * unavailable"). Returns `null` when there is nothing to show.
 */
export function formatDisplayName(user?: string | null): string | null {
  const raw = (user ?? "").trim();
  if (!raw) return null;
  if (isEmail(raw)) return raw; // no display-name claim → show the email
  const parts = raw.split(/\s+/).filter(Boolean);
  if (parts.length <= 1) return parts[0] ?? raw;
  const first = parts[0];
  const last = parts[parts.length - 1];
  return `${first.charAt(0).toUpperCase()}. ${last}`;
}

/** Two-letter avatar initials from a name ("Joshua Hall" → "JH") or email. */
export function displayInitials(user?: string | null): string {
  const raw = (user ?? "").trim();
  if (!raw) return "–";
  if (isEmail(raw)) return raw.slice(0, 2).toUpperCase();
  const parts = raw.split(/\s+/).filter(Boolean);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
}

/** Role sub-label shown under the name. */
export function roleLabel(identity: UserIdentity | null): string | null {
  if (!identity?.authenticated) return null;
  if (identity.isOperator) return "Operator";
  if (identity.roles?.length) return "Client";
  return null;
}

/**
 * Whether settings / admin controls should be visible. Client users get a
 * read-only MI surface (no settings/admin); operators/admins may see settings.
 * Defaults to HIDDEN when identity is unknown (fail-closed for a client build).
 */
export function canSeeAdminControls(identity: UserIdentity | null): boolean {
  return !!identity?.isOperator;
}
