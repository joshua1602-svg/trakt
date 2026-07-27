#!/usr/bin/env node
/**
 * Build-output scan.
 *
 * Greps the artefacts that actually ship — the standalone server bundle, the
 * client bundle and the committed demo pack — for secrets and for internal
 * references that must never reach a public deployment.
 *
 * This runs after `next build` and before deploy. It is deliberately a scan of
 * *output*, not of source: source review catches intent, but only the built
 * artefact proves what a visitor can actually download.
 *
 *   node scripts/scan-build.mjs
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");

const TARGETS = [
  { dir: ".next/standalone", label: "server bundle", clientVisible: false },
  { dir: ".next/static", label: "client bundle", clientVisible: true },
];
// The pack is imported server-side only (`lib/demo-pack.ts` is `server-only`)
// and is never emitted into the client bundle — verified by the client-bundle
// scan below, which would flag it if it were.
const FILES = [{ file: "data/demo-pack.json", label: "demo pack", clientVisible: false }];

const SCANNABLE = new Set([".js", ".mjs", ".cjs", ".json", ".css", ".html", ".txt", ".map"]);

/** Secret shapes. Each must be specific enough not to fire on ordinary code. */
const SECRET_PATTERNS = [
  { name: "Azure Storage account key", re: /AccountKey=[A-Za-z0-9+/=]{40,}/ },
  { name: "Azure connection string", re: /DefaultEndpointsProtocol=https?;AccountName=/ },
  { name: "App Insights connection string", re: /InstrumentationKey=[0-9a-f-]{36}/i },
  { name: "AWS access key id", re: /\bAKIA[0-9A-Z]{16}\b/ },
  { name: "Resend API key", re: /\bre_[A-Za-z0-9]{20,}\b/ },
  { name: "SendGrid API key", re: /\bSG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b/ },
  { name: "Slack token", re: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/ },
  { name: "GitHub token", re: /\bgh[pousr]_[A-Za-z0-9]{30,}\b/ },
  { name: "private key block", re: /-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----/ },
  { name: "JWT", re: /\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\./ },
  { name: "bearer literal", re: /Bearer\s+[A-Za-z0-9._-]{30,}/ },
];

/**
 * Internal references that must not appear ANYWHERE in a build output.
 *
 * Each of these would mean the landing page has reached into Trakt
 * infrastructure — the thing this deployment is architected not to do. Their
 * presence in the server bundle is as much a defect as in the client one.
 */
const PROHIBITED_ANYWHERE = [
  { name: "Azure blob endpoint", re: /[a-z0-9-]+\.blob\.core\.windows\.net/i },
  { name: "internal MI API host", re: /trakt-mi-api[a-z0-9-]*\.azurewebsites\.net/i },
  { name: "internal storage layout", re: /processed-v2\// },
  { name: "MI agent env var", re: /\bMI_AGENT_[A-Z_]+/ },
  { name: "Copilot env var", re: /\bTRAKT_COPILOT_[A-Z_]+/ },
];

/**
 * References that are acceptable server-side but must never reach a browser.
 *
 * Next's standalone output records its own build directory, and the demo
 * pack's server-only provenance block names the dataset it was generated from
 * — both are legitimate and neither is downloadable. In the client bundle the
 * same strings would be an information leak, so they stay errors there.
 */
const PROHIBITED_CLIENT_ONLY = [
  { name: "repository absolute path", re: /\/home\/[a-z0-9_-]+\/trakt/i },
  { name: "synthetic source path", re: /synthetic_demo\/output\// },
  { name: "engine module reference", re: /mi_agent(_api)?\.[a-z_]+/ },
];

/** Secrets that must never be inlined into anything the browser receives. */
const CLIENT_FORBIDDEN_ENV = [
  "DEMO_SESSION_SECRET",
  "EMAIL_API_KEY",
  "CRM_WEBHOOK_SECRET",
  "APPLICATIONINSIGHTS_CONNECTION_STRING",
  "RATE_LIMIT_STORE_URL",
];

function* walk(dir) {
  let entries;
  try {
    entries = readdirSync(dir);
  } catch {
    return;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry);
    let info;
    try {
      info = statSync(full);
    } catch {
      continue;
    }
    if (info.isDirectory()) {
      if (entry === "node_modules") {
        // The server bundle vendors dependencies; their source is not ours to
        // police, and scanning it produces only noise.
        continue;
      }
      yield* walk(full);
    } else if (SCANNABLE.has(path.extname(entry))) {
      yield full;
    }
  }
}

const findings = [];

function scanText(text, label, rel, clientVisible) {
  for (const { name, re } of SECRET_PATTERNS) {
    if (re.test(text)) findings.push(`[secret] ${name} in ${label}: ${rel}`);
  }
  for (const { name, re } of PROHIBITED_ANYWHERE) {
    if (re.test(text)) findings.push(`[internal] ${name} in ${label}: ${rel}`);
  }
  if (clientVisible) {
    for (const { name, re } of PROHIBITED_CLIENT_ONLY) {
      if (re.test(text)) findings.push(`[leak] ${name} in ${label}: ${rel}`);
    }
    for (const key of CLIENT_FORBIDDEN_ENV) {
      // The literal value would appear beside its name only if something
      // serialised the environment into the bundle.
      if (new RegExp(`${key}\\s*[:=]\\s*["'][^"']{8,}`).test(text)) {
        findings.push(`[leak] ${key} value inlined in ${label}: ${rel}`);
      }
    }
  }
}

let scanned = 0;

for (const { dir, label, clientVisible } of TARGETS) {
  const abs = path.join(ROOT, dir);
  for (const file of walk(abs)) {
    scanned += 1;
    scanText(readFileSync(file, "utf8"), label, path.relative(ROOT, file), clientVisible);
  }
}

for (const { file, label, clientVisible } of FILES) {
  const abs = path.join(ROOT, file);
  try {
    scanned += 1;
    scanText(readFileSync(abs, "utf8"), label, file, clientVisible);
  } catch {
    findings.push(`[missing] ${label} not found at ${file}`);
  }
}

if (scanned === 0) {
  console.error("scan-build: nothing scanned — run `npm run build` first.");
  process.exit(1);
}

if (findings.length > 0) {
  console.error(`scan-build: ${findings.length} finding(s) across ${scanned} files:\n`);
  for (const finding of new Set(findings)) console.error(`  ${finding}`);
  process.exit(1);
}

console.log(`scan-build: clean — ${scanned} files, no secrets or internal references.`);
