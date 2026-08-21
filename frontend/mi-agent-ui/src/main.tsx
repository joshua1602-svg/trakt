import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import { AuthBoundary } from "./auth/AuthBoundary";
import { bootstrapAuth } from "./auth/bootstrap";
import { resolveAuthConfig } from "./auth/msalConfig";
import "./index.css";

const root = createRoot(document.getElementById("root")!);

function render(msal: Parameters<typeof AuthBoundary>[0]["msal"] = null) {
  root.render(
    <StrictMode>
      <AuthBoundary msal={msal}>
        <App />
      </AuthBoundary>
    </StrictMode>,
  );
}

const authConfig = resolveAuthConfig();

if (!authConfig.enabled) {
  // VITE_MI_BEARER_AUTH off: render immediately, exactly as before. MSAL is
  // never constructed and no token provider is registered, so the Static Web
  // Apps path is untouched.
  render();
} else {
  // Flag on: MSAL must consume any redirect fragment and settle on an account
  // BEFORE the first render, or the app renders signed-out, redirects to
  // Microsoft again and loops. Rendering after the promise resolves is what
  // makes the sign-in land.
  void bootstrapAuth(authConfig).then(({ msal }) => render(msal));
}
