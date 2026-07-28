#!/usr/bin/env node
/**
 * Post-deploy portfolio identity verification.
 *
 * Asks the running deployment which demo pack it is serving, and compares the
 * answer field-by-field against the pack committed to this repository. A
 * difference on any field means the App Service is running a bundle built from
 * a different dataset than the one that passed the verify job — which is a
 * failed deployment, not a warning.
 *
 *   node scripts/verify-deployed-identity.mjs --base https://example.net
 *   node scripts/verify-deployed-identity.mjs --base http://127.0.0.1:3000 \
 *       --pack data/demo-pack.json
 *
 * Fails closed on every axis: a differing field, a missing field, an extra
 * field, a non-JSON body, a non-200 status, or an endpoint that cannot be
 * reached at all. There is no "could not check, carrying on" outcome — an
 * unverifiable deployment is treated exactly like a wrong one.
 *
 * Why this and not a grep of the homepage: the previous check searched the
 * served HTML for the raw portfolio id. That worked only because the id
 * travelled inside Next's serialised RSC payload as an unrendered prop — an
 * artefact of framework serialisation, not a published fact, and one that
 * changes shape whenever the component tree does. The homepage marker is still
 * cross-checked below, but as corroboration of a contract, not as the contract.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");

/**
 * The identity document's exact field set.
 *
 * Mirrors `DEMO_IDENTITY_FIELDS` in `src/lib/demo-identity.ts`. The two are
 * held together by `tests/api-demo-identity.test.ts`, which builds the expected
 * document both ways and fails if they differ — so this list cannot silently
 * fall behind the endpoint it is checking.
 */
export const IDENTITY_FIELDS = [
  "clientId",
  "portfolioId",
  "sourceFingerprint",
  "reportingDate",
  "currency",
  "totalBalanceDisplay",
  "platformBalanceDisplay",
  "exposures",
  "packVersion",
  "synthetic",
];

/** Project a committed demo pack onto the identity document it should serve. */
export function expectedIdentityFromPack(pack) {
  const source = pack?.source ?? {};
  const portfolio = pack?.portfolio ?? {};
  return {
    clientId: source.clientId,
    portfolioId: source.portfolioId,
    sourceFingerprint: source.canonicalSha256,
    reportingDate: source.reportingDate,
    currency: source.currency,
    totalBalanceDisplay: portfolio.totalBalanceDisplay,
    platformBalanceDisplay: portfolio.platformBalanceDisplay,
    exposures: source.exposures,
    packVersion: pack?.packVersion,
    synthetic: true,
  };
}

/**
 * Compare a served document against the expected one.
 *
 * Returns every problem found rather than the first, so one failed run names
 * the whole discrepancy instead of revealing it a deploy at a time.
 *
 * Comparison is strict: same type and same value. `"118"` is not `118`, because
 * a served document whose types have drifted is a different contract even when
 * it renders the same.
 */
export function compareIdentity(expected, served) {
  const problems = [];

  if (served === null || typeof served !== "object" || Array.isArray(served)) {
    return { ok: false, problems: ["the identity document is not a JSON object"] };
  }

  const servedKeys = Object.keys(served).sort();
  const knownKeys = [...IDENTITY_FIELDS].sort();

  const missing = knownKeys.filter((k) => !servedKeys.includes(k));
  if (missing.length > 0) {
    problems.push(`missing field(s): ${missing.join(", ")}`);
  }

  // An unexpected field is a mismatch, not a curiosity. This endpoint's whole
  // value is that its contents are exhaustively known; a field nobody reviewed
  // is a field nobody reviewed for sensitivity either.
  const unexpected = servedKeys.filter((k) => !knownKeys.includes(k));
  if (unexpected.length > 0) {
    problems.push(`unexpected field(s): ${unexpected.join(", ")}`);
  }

  for (const field of IDENTITY_FIELDS) {
    if (missing.includes(field)) continue;
    const want = expected[field];
    const got = served[field];
    if (typeof got !== typeof want) {
      problems.push(
        `${field}: type ${typeof got} differs from expected ${typeof want} ` +
          `(${JSON.stringify(got)} vs ${JSON.stringify(want)})`,
      );
      continue;
    }
    if (got !== want) {
      problems.push(
        `${field}: served ${JSON.stringify(got)} differs from committed ` +
          `${JSON.stringify(want)}`,
      );
    }
  }

  return { ok: problems.length === 0, problems };
}

/** Pull the `trakt:pack` marker out of served HTML, or null if it is absent. */
export function markerFromHtml(html) {
  const match = /<meta[^>]+name="trakt:pack"[^>]+content="([^"]*)"/.exec(html ?? "");
  return match ? match[1] : null;
}

/** The marker the identity document implies. Same three parts, same order. */
export function markerFromIdentity(identity) {
  return [identity.clientId, identity.portfolioId, identity.reportingDate].join("/");
}

/* -------------------------------------------------------------------------- */
/* CLI                                                                        */
/* -------------------------------------------------------------------------- */

const TIMEOUT_MS = 15_000;
const ATTEMPTS = 3;
const RETRY_DELAY_MS = 10_000;

function fail(message, detail = []) {
  console.error(`::error::${message}`);
  for (const line of detail) console.error(`  ${line}`);
  process.exitCode = 1;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Fetch a URL, retrying only what is plausibly transient.
 *
 * An App Service that has just been restarted can refuse a connection or return
 * 502 for a few seconds. A 404 cannot be transient — it means the route is not
 * in the deployed bundle — so it is returned immediately rather than waited on.
 */
async function fetchWithRetry(url, { accept }) {
  let last = { ok: false, reason: "not attempted" };

  for (let attempt = 1; attempt <= ATTEMPTS; attempt += 1) {
    let response;
    try {
      response = await fetch(url, {
        headers: { Accept: accept },
        redirect: "follow",
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
    } catch (error) {
      last = { ok: false, reason: `request failed: ${error?.message ?? error}` };
      if (attempt < ATTEMPTS) {
        console.log(`  attempt ${attempt}: ${last.reason} — retrying`);
        await sleep(RETRY_DELAY_MS);
      }
      continue;
    }

    const body = await response.text();
    if (response.ok) {
      return {
        ok: true,
        status: response.status,
        contentType: response.headers.get("content-type") ?? "",
        body,
      };
    }

    last = {
      ok: false,
      status: response.status,
      reason: `HTTP ${response.status}`,
      body,
    };
    const transient = response.status >= 500 || response.status === 429;
    if (!transient || attempt === ATTEMPTS) return last;

    console.log(`  attempt ${attempt}: ${last.reason} — retrying`);
    await sleep(RETRY_DELAY_MS);
  }

  return last;
}

function parseArgs(argv) {
  const args = { base: "", pack: path.join(ROOT, "data", "demo-pack.json") };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--base") args.base = argv[i + 1] ?? "";
    if (argv[i] === "--pack") args.pack = path.resolve(argv[i + 1] ?? "");
  }
  return args;
}

async function main(argv) {
  const { base, pack: packPath } = parseArgs(argv);

  if (!base) {
    fail("no --base origin given; nothing to verify against");
    return;
  }
  const origin = base.replace(/\/+$/, "");

  let pack;
  try {
    pack = JSON.parse(readFileSync(packPath, "utf8"));
  } catch (error) {
    fail(`the committed demo pack could not be read: ${error?.message ?? error}`, [
      `path: ${packPath}`,
    ]);
    return;
  }

  const expected = expectedIdentityFromPack(pack);
  console.log("expected identity, from the committed pack:");
  for (const field of IDENTITY_FIELDS) {
    console.log(`  ${field.padEnd(22)} ${JSON.stringify(expected[field])}`);
  }
  console.log("");

  /* --- the identity document ------------------------------------------- */

  const url = `${origin}/api/demo-identity`;
  console.log(`GET ${url}`);
  const response = await fetchWithRetry(url, { accept: "application/json" });

  if (!response.ok) {
    fail(`/api/demo-identity is unavailable — ${response.reason}`, [
      response.status === 404
        ? "The route is not present in the deployed bundle: App Service is " +
          "serving a build from before this control existed."
        : "The deployment could not be verified, so it is treated as failed.",
      ...(response.body ? [`body: ${response.body.slice(0, 400)}`] : []),
    ]);
    return;
  }

  if (!response.contentType.toLowerCase().includes("application/json")) {
    fail("/api/demo-identity did not return JSON", [
      `content-type: ${response.contentType || "(none)"}`,
      `body: ${response.body.slice(0, 400)}`,
    ]);
    return;
  }

  let served;
  try {
    served = JSON.parse(response.body);
  } catch (error) {
    fail(`/api/demo-identity returned a malformed body: ${error?.message ?? error}`, [
      `body: ${response.body.slice(0, 400)}`,
    ]);
    return;
  }

  const { ok, problems } = compareIdentity(expected, served);
  if (!ok) {
    fail("the deployment is serving a different demo pack than the committed one", [
      ...problems,
      "",
      "This is a deployment problem, not a build one — the committed pack",
      "passed reproducibility and source-identity checks in the verify job.",
      "Do not regenerate the pack to make this pass.",
    ]);
    return;
  }
  console.log("  identity document matches the committed pack on every field");

  /* --- corroboration: the homepage marker ------------------------------ */

  console.log(`GET ${origin}/`);
  const page = await fetchWithRetry(`${origin}/`, { accept: "text/html" });
  if (!page.ok) {
    fail(`the homepage is unreachable — ${page.reason}`);
    return;
  }

  const marker = markerFromHtml(page.body);
  const expectedMarker = markerFromIdentity(expected);

  if (marker === null) {
    fail("the served homepage carries no trakt:pack marker", [
      "The identity endpoint agreed with the committed pack, so the HTML and",
      "the API are being served by different bundles, or the marker was",
      "removed from src/app/layout.tsx.",
    ]);
    return;
  }
  if (marker !== expectedMarker) {
    fail("the served homepage and the identity endpoint disagree", [
      `homepage marker: ${marker}`,
      `expected:        ${expectedMarker}`,
      "A partially-updated deployment, or a cached HTML response from an",
      "older bundle.",
    ]);
    return;
  }

  if (!page.body.includes(expected.totalBalanceDisplay)) {
    fail(
      `the served homepage does not display the pinned total ` +
        `${expected.totalBalanceDisplay}`,
      ["The identity endpoint is correct, so the page is rendering from", "different data."],
    );
    return;
  }

  console.log(`  homepage marker agrees: ${marker}`);
  console.log(`  homepage displays the pinned total: ${expected.totalBalanceDisplay}`);
  console.log("");
  console.log(
    `verified: ${expected.clientId} / ${expected.portfolioId} @ ` +
      `${expected.reportingDate}, fingerprint ${expected.sourceFingerprint.slice(0, 12)}…`,
  );
}

// Only when run directly. Importing this module for its pure helpers, as the
// test suite does, must not fire a network request.
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(import.meta.filename)) {
  await main(process.argv.slice(2));
}
