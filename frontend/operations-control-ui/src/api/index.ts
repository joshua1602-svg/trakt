import { HttpOpsClient } from "./HttpOpsClient";
import { MockOpsClient } from "./MockOpsClient";
import type { OpsClient } from "./OpsClient";

export function isMockMode(): boolean {
  return import.meta.env.VITE_OPS_MODE === "mock";
}

export function createOpsClient(): OpsClient {
  return isMockMode() ? new MockOpsClient() : new HttpOpsClient();
}

export { HttpOpsClient } from "./HttpOpsClient";
export { MockOpsClient } from "./MockOpsClient";
export { OpsError } from "./OpsClient";
export type { OpsClient } from "./OpsClient";
export * from "./types";
