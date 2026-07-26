"use client";

import { useEffect } from "react";

import { captureAttribution } from "@/lib/analytics";

/**
 * Records the campaign a visitor arrived on, once, on arrival.
 *
 * Mounted high in the page rather than inside the lead form: a visitor who
 * lands from a campaign and submits the form has to be attributed even though
 * the query string is long gone from their attention by then, and a visitor
 * who never scrolls that far still deserves to be attributed correctly if they
 * return later in the same session.
 *
 * Renders nothing.
 */
export function AttributionCapture() {
  useEffect(() => {
    captureAttribution();
  }, []);

  return null;
}
