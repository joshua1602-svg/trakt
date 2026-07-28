/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_OPS_API_URL?: string;
  readonly VITE_OPS_MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
