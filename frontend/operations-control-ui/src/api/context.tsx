import { createContext, useContext, type ReactNode } from "react";
import type { OpsClient } from "./OpsClient";

const OpsClientContext = createContext<OpsClient | null>(null);

export function OpsClientProvider({
  client,
  children,
}: {
  client: OpsClient;
  children: ReactNode;
}) {
  return <OpsClientContext.Provider value={client}>{children}</OpsClientContext.Provider>;
}

export function useOpsClient(): OpsClient {
  const client = useContext(OpsClientContext);
  if (!client) {
    throw new Error("OpsClientProvider is missing above this component");
  }
  return client;
}
