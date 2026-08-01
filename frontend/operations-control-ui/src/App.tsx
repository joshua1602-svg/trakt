import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import { createOpsClient, isMockMode } from "@/api";
import { OpsClientProvider } from "@/api/context";
import { SessionProvider } from "@/api/session";
import { RouteErrorBoundary } from "@/components/ErrorBoundary";
import { Shell } from "@/components/Shell";
import { SignIn } from "@/components/SignIn";
import { ToastProvider } from "@/components/Toast";
import { getToken, saveToken, UNAUTHORIZED_EVENT } from "@/lib/token";
import { BatchDetailScreen } from "@/screens/BatchDetail";
import { HistoryScreen } from "@/screens/History";
import { HomeScreen } from "@/screens/Home";
import { NewWorkflowScreen } from "@/screens/NewWorkflow";
import { ReviewDetailScreen } from "@/screens/ReviewDetail";
import { ReviewsScreen } from "@/screens/Reviews";
import { RulesScreen } from "@/screens/Rules";
import { WorkflowDetailScreen } from "@/screens/WorkflowDetail";
import { WorkflowsScreen } from "@/screens/Workflows";
import { AgentCaseScreen } from "@/screens/agent/AgentCase";
import { AgentCasesScreen } from "@/screens/agent/AgentCases";
import { AgentOnly } from "@/screens/agent/AgentLayout";
import { OnboardingCaseScreen } from "@/screens/onboarding/CaseWizard";
import { OnboardingClientScreen } from "@/screens/onboarding/ClientView";
import {
  ConcentrationClientsScreen,
  ConcentrationReviewScreen,
} from "@/screens/concentration/ConcentrationReview";
import { OnboardingHomeScreen } from "@/screens/onboarding/Home";
import { AdminOnly } from "@/screens/admin/AdminLayout";
import { ConfigAssetsScreen } from "@/screens/admin/ConfigAssets";
import { ConfigHistoryScreen } from "@/screens/admin/ConfigHistory";
import { ConfigOverviewScreen } from "@/screens/admin/ConfigOverview";
import { ConfigRegimesScreen } from "@/screens/admin/ConfigRegimes";
import { ConfigSystemScreen } from "@/screens/admin/ConfigSystem";

/** Shows the sign-in card until an access key is stored; re-shows it on 401. */
function AuthGate({ children }: { children: ReactNode }) {
  const [signedIn, setSignedIn] = useState(() => isMockMode() || Boolean(getToken()));

  useEffect(() => {
    const onUnauthorized = () => setSignedIn(false);
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
  }, []);

  if (!signedIn) {
    return (
      <SignIn
        onSubmit={(token) => {
          saveToken(token);
          setSignedIn(true);
        }}
      />
    );
  }
  return <>{children}</>;
}

export default function App() {
  const client = useMemo(() => createOpsClient(), []);

  return (
    <OpsClientProvider client={client}>
      <ToastProvider>
        <AuthGate>
          {/* SessionProvider loads the signed-in principal once. The Shell needs
              it to decide whether to offer the administrator entry, and AdminOnly
              needs it to decide what to render — so it must wrap both. */}
          <SessionProvider>
            <Shell>
              <Routes>
                <Route path="/" element={<HomeScreen />} />
                <Route path="/new" element={<NewWorkflowScreen />} />
                <Route path="/batches/:id" element={<BatchDetailScreen />} />
                <Route path="/workflows" element={<WorkflowsScreen />} />
                <Route path="/workflows/:id" element={<WorkflowDetailScreen />} />
                <Route path="/reviews" element={<ReviewsScreen />} />
                <Route path="/reviews/:id" element={<ReviewDetailScreen />} />
                <Route path="/rules" element={<RulesScreen />} />
                <Route path="/history" element={<HistoryScreen />} />
                {/* Client Onboarding — standing configuration. It sits beside
                    Operations: Operations processes deliveries, Onboarding
                    creates the configuration those deliveries resolve with. */}
                <Route path="/concentration" element={<ConcentrationClientsScreen />} />
          <Route path="/concentration/:client" element={<ConcentrationReviewScreen />} />
          <Route path="/onboarding" element={<OnboardingHomeScreen />} />
                <Route path="/onboarding/cases/:id" element={<OnboardingCaseScreen />} />
                <Route path="/onboarding/clients/:id" element={<OnboardingClientScreen />} />
                {/* The OCC Agent. Flag-gated, synthetic behind the guard — the
                    backend refuses these routes outright when the same flag is
                    off in its own environment — and wrapped in a route-level
                    error boundary so a rendering fault shows a card and keeps
                    the OCC shell standing instead of blanking the page. */}
                <Route
                  path="/agent"
                  element={
                    <AgentOnly>
                      <RouteErrorBoundary>
                        <AgentCasesScreen />
                      </RouteErrorBoundary>
                    </AgentOnly>
                  }
                />
                <Route
                  path="/agent/:caseId"
                  element={
                    <AgentOnly>
                      <RouteErrorBoundary>
                        <AgentCaseScreen />
                      </RouteErrorBoundary>
                    </AgentOnly>
                  }
                />
                {/* Administrator area. The guard decides what to render; the
                    backend decides what is allowed on every request. */}
                <Route
                  path="/admin/config"
                  element={
                    <AdminOnly>
                      <ConfigOverviewScreen />
                    </AdminOnly>
                  }
                />
                <Route
                  path="/admin/config/system"
                  element={
                    <AdminOnly>
                      <ConfigSystemScreen />
                    </AdminOnly>
                  }
                />
                <Route
                  path="/admin/config/assets"
                  element={
                    <AdminOnly>
                      <ConfigAssetsScreen />
                    </AdminOnly>
                  }
                />
                <Route
                  path="/admin/config/regimes"
                  element={
                    <AdminOnly>
                      <ConfigRegimesScreen />
                    </AdminOnly>
                  }
                />
                <Route
                  path="/admin/config/history"
                  element={
                    <AdminOnly>
                      <ConfigHistoryScreen />
                    </AdminOnly>
                  }
                />
              </Routes>
            </Shell>
          </SessionProvider>
        </AuthGate>
      </ToastProvider>
    </OpsClientProvider>
  );
}
