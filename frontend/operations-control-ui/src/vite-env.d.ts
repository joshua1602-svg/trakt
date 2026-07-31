/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_OPS_API_URL?: string;
  readonly VITE_OPS_MODE?: string;
  /** Shows the OCC Agent tab. Off unless explicitly set — see lib/features.ts. */
  readonly VITE_OCC_AGENT_SYNTHETIC_ENABLED?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
