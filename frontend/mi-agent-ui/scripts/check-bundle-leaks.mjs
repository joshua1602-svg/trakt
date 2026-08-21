#!/usr/bin/env node
/**
 * Fail the build if the compiled bundle carries a credential.
 *
 * Vite inlines every `VITE_*` value into the output, so this is not a
 * theoretical risk: one mis-named workflow variable (`VITE_AAD_CLIENT_SECRET=…`)
 * would put a working client secret on a public CDN, and nothing else in the
 * pipeline would notice. The SPA is a public client using Authorization Code +
 * PKCE and needs no secret, so anything secret-shaped in `dist/` is a mistake by
 * definition.
 *
 * Usage:  node scripts/check-bundle-leaks.mjs [dist-dir]
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const dist = process.argv[2] ?? "dist";

/** Every file under `dir`, recursively. */
function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...walk(path));
    else out.push(path);
  }
  return out;
}

// Patterns that indicate a credential rather than a configuration value. A
// client ID, an authority URL and an `api://…/MI.Access` scope are all PUBLIC
// identifiers and are expected to appear.
//
// These match a secret with a VALUE, not the mere mention of one. msal-browser
// ships a table of OAuth parameter NAMES — "post_logout_redirect_uri",
// "id_token_hint", "client_secret", "client_assertion" — because it has to
// build requests that use them. A scan that flags the bare string
// `"client_secret"` fails on every honest bundle, and a check that always fails
// gets switched off, which is worse than not having one.
const FORBIDDEN = [
  [/client_secret\s*[:=]\s*["'][^"']+["']/i, "an assigned client_secret value"],
  [/clientSecret\s*[:=]\s*["'][^"']+["']/, "an assigned clientSecret value"],
  [/client_secret=[A-Za-z0-9._~-]{8,}/, "a client secret in a query string"],
  [/VITE_[A-Z_]*(SECRET|PASSWORD|PRIVATE)/, "a secret-shaped build variable"],
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----/, "a private key"],
  [/\bAccountKey=[A-Za-z0-9+/=]{20,}/, "a storage account key"],
  [/\bSharedAccessSignature\b|\bsig=[A-Za-z0-9%]{20,}/, "a SAS token"],
];

let failures = 0;
let scanned = 0;

for (const file of walk(dist)) {
  if (!/\.(js|mjs|cjs|css|html|map|json)$/.test(file)) continue;
  scanned += 1;
  const text = readFileSync(file, "utf8");
  for (const [pattern, what] of FORBIDDEN) {
    if (pattern.test(text)) {
      console.error(`FAIL: ${file} contains ${what} (${pattern})`);
      failures += 1;
    }
  }
}

if (scanned === 0) {
  console.error(`FAIL: no bundle files found under ${dist}/ — did the build run?`);
  process.exit(1);
}

if (failures > 0) {
  console.error(`\n${failures} secret-shaped match(es) in the bundle. A SPA bundle is public.`);
  process.exit(1);
}

console.log(`Bundle secret scan OK: ${scanned} file(s) in ${dist}/, no credential material.`);
