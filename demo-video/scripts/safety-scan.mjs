/**
 * Pre-render safety gate for the video project.
 *
 * The authoritative scan lives in `demo_platform/safety.py`, which reads the
 * prohibited-string list live out of the production client configs. This script
 * is the render-time guard: it refuses to render unless
 *
 *   1. the fixtures are present and carry the synthetic-data banner;
 *   2. the Python safety report is present and passed;
 *   3. the Python assertion report is present and passed;
 *   4. a direct re-scan of the bundled fixtures and every source file finds no
 *      prohibited string, email address, GUID, storage endpoint, connection
 *      string, SAS URL or bearer token.
 *
 * Point 4 is deliberately redundant with the Python scan: the fixtures are copied
 * into `public/fixtures` and the scene code is written by hand, so the thing that
 * actually gets bundled is re-checked here, at the last moment before rendering.
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const FIXTURES = join(ROOT, "public", "fixtures");

const SYNTHETIC_BANNER = "SYNTHETIC DEMONSTRATION DATA";

/** Patterns that must not appear in anything that gets bundled. */
const PATTERNS = [
  {
    name: "email_address",
    re: /\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b/g,
    why: "a live email address must never appear in demonstration material",
  },
  {
    name: "guid_or_tenant_id",
    re: /\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b/g,
    why: "a GUID may be a real tenant, subscription or application identifier",
  },
  {
    name: "azure_storage_endpoint",
    re: /[A-Za-z0-9-]+\.(?:blob|table|queue|file)\.core\.windows\.net/g,
    why: "a real Azure storage endpoint reveals a production storage account",
  },
  {
    name: "azure_web_host",
    re: /[A-Za-z0-9-]+\.(?:azurewebsites|azurestaticapps)\.net/g,
    why: "a real Azure hostname reveals a production deployment",
  },
  {
    name: "storage_connection_string",
    re: /DefaultEndpointsProtocol=|AccountKey=|SharedAccessSignature=/g,
    why: "a storage connection string or account key is a credential",
  },
  {
    name: "sas_signed_url",
    re: /[?&](?:sig|sv|se|st|sp)=[^\s"'&]+/g,
    why: "a signed (SAS) URL is a bearer credential",
  },
  {
    name: "bearer_token",
    re: /\bBearer\s+[A-Za-z0-9\-._~+/]{20,}/g,
    why: "a bearer token is a credential",
  },
  {
    name: "jwt",
    re: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g,
    why: "a JWT is a credential",
  },
  {
    name: "live_download_token",
    re: /artifacts\/download\?token=(?!<redacted>)[A-Za-z0-9._-]{12,}/g,
    why: "a live artefact download token is a short-lived credential",
  },
];

/** Files that legitimately contain the patterns because they define them. */
const SELF_REFERENTIAL = new Set(["safety-scan.mjs", "safety_report.json"]);

const TEXT_EXTS = new Set([
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".mjs",
  ".json",
  ".md",
  ".css",
  ".html",
  ".srt",
  ".vtt",
  ".txt",
]);

const walk = (dir, acc = []) => {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return acc;
  }
  for (const entry of entries) {
    if (entry.name === "node_modules" || entry.name === "out" || entry.name.startsWith(".")) {
      continue;
    }
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full, acc);
    } else if (TEXT_EXTS.has(extname(entry.name)) && !SELF_REFERENTIAL.has(entry.name)) {
      acc.push(full);
    }
  }
  return acc;
};

const readJson = (path) => {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
};

const failures = [];
const notes = [];

// 1 — the fixtures must exist.
const metricsPath = join(FIXTURES, "demo_metrics.json");
const metrics = readJson(metricsPath);
if (!metrics) {
  failures.push(
    `fixtures missing: ${relative(ROOT, metricsPath)}. Run:\n` +
      "    python -m demo_platform.run_demo --all",
  );
} else if (!String(metrics.synthetic_notice ?? "").includes(SYNTHETIC_BANNER)) {
  failures.push(
    "demo_metrics.json does not carry the synthetic-data banner — refusing to " +
      "render material that is not marked as synthetic",
  );
} else {
  notes.push(`fixtures present and marked synthetic (${metrics.synthetic_notice})`);
}

// 2 — the Python safety report must have passed.
const safety = readJson(join(FIXTURES, "safety_report.json"));
if (!safety) {
  failures.push(
    "safety_report.json missing — run `python -m demo_platform.run_demo --safety`",
  );
} else if (safety.ok !== true) {
  failures.push(
    `the Python safety scan reported ${safety.findingCount} finding(s); ` +
      "the render is blocked until they are resolved",
  );
} else {
  notes.push(
    `Python safety scan passed (${safety.filesScanned} files, ` +
      `${safety.prohibitedStrings?.length ?? 0} prohibited strings)`,
  );
}

// 3 — the reconciliation gate must have passed.
const assertions = readJson(join(FIXTURES, "assertion_report.json"));
if (!assertions) {
  failures.push(
    "assertion_report.json missing — run `python -m demo_platform.run_demo --assert`",
  );
} else if (assertions.ok !== true) {
  failures.push(
    `the reconciliation gate failed (${assertions.checksFailed} of ` +
      `${assertions.checksRun} checks) — the film would state figures the data ` +
      "does not support",
  );
} else {
  notes.push(
    `reconciliation gate passed (${assertions.checksPassed}/${assertions.checksRun} checks)`,
  );
}

// 4 — re-scan everything that gets bundled.
const prohibited = Array.isArray(safety?.prohibitedStrings) ? safety.prohibitedStrings : [];
const files = [...walk(join(ROOT, "src")), ...walk(FIXTURES), ...walk(join(ROOT, "scripts"))];
let scanned = 0;

for (const file of files) {
  let text;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    continue;
  }
  scanned += 1;
  const rel = relative(ROOT, file);
  const lines = text.split("\n");

  lines.forEach((line, i) => {
    const lower = line.toLowerCase();
    for (const needle of prohibited) {
      const idx = lower.indexOf(needle.toLowerCase());
      if (idx === -1) continue;
      const before = idx > 0 ? lower[idx - 1] : " ";
      const afterIdx = idx + needle.length;
      const after = afterIdx < lower.length ? lower[afterIdx] : " ";
      if (/[a-z0-9]/.test(before) || /[a-z0-9]/.test(after)) continue;
      failures.push(
        `prohibited client string "${needle}" in ${rel}:${i + 1}`,
      );
    }
    for (const pattern of PATTERNS) {
      pattern.re.lastIndex = 0;
      const match = pattern.re.exec(line);
      if (match) {
        failures.push(`${pattern.name} in ${rel}:${i + 1} — ${pattern.why}`);
      }
    }
  });
}
notes.push(`re-scanned ${scanned} bundled files against ${PATTERNS.length} patterns`);

// Report.
console.log("[safety] pre-render gate");
for (const note of notes) console.log(`  ok   ${note}`);
if (failures.length) {
  console.error(`\n[safety] BLOCKED — ${failures.length} finding(s):`);
  for (const f of failures.slice(0, 30)) console.error(`  fail ${f}`);
  if (failures.length > 30) console.error(`  … and ${failures.length - 30} more`);
  process.exit(2);
}
console.log("[safety] PASS — safe to render");
