import type { Metadata, Viewport } from "next";

import { demoPack } from "@/lib/demo-pack";
import { publicConfig } from "@/lib/public-config";

import "./globals.css";

const TITLE = "Trakt | Governed Portfolio Intelligence";
const DESCRIPTION =
  "Trakt transforms fragmented portfolio data into trusted analytics, automated " +
  "reporting and governed workflows across Copilot, Teams and the Trakt workspace.";

export const metadata: Metadata = {
  metadataBase: new URL(publicConfig.siteUrl),
  title: { default: TITLE, template: "%s | Trakt" },
  description: DESCRIPTION,
  applicationName: "Trakt",
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    siteName: "Trakt",
    title: TITLE,
    description: DESCRIPTION,
    url: "/",
    locale: "en_GB",
    images: [{ url: "/opengraph-image", width: 1200, height: 630, alt: TITLE }],
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    images: ["/opengraph-image"],
  },
  robots: { index: true, follow: true },
  category: "technology",
  other: {
    // Provenance of the demo pack this build was compiled against, published
    // deliberately so the post-deploy smoke check has a stable contract.
    //
    // That check previously grepped the served HTML for the raw portfolio id.
    // It worked only because the id travels inside Next's serialised RSC
    // payload as an unrendered prop — an accident of framework serialisation,
    // not a published fact. Dropping that prop, or a change in how Next
    // serialises it, would have broken the deploy gate with no code change and
    // no obvious cause.
    //
    // A meta tag rather than visible copy: an internal portfolio identifier is
    // provenance, not something a prospect should be reading.
    "trakt:pack": [
      demoPack.source.clientId,
      demoPack.source.portfolioId,
      demoPack.source.reportingDate,
    ].join("/"),
  },
};

export const viewport: Viewport = {
  themeColor: "#0c1024",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
};

/**
 * Structured data.
 *
 * Deliberately limited to what is verifiable from this page: the organisation
 * and the software product it publishes. No ratings, no pricing, no address,
 * no claims the site cannot support.
 */
const structuredData = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": `${publicConfig.siteUrl}/#organization`,
      name: "Trakt",
      url: publicConfig.siteUrl,
      description:
        "Governed portfolio operating, analytics and reporting layer for specialist lending.",
    },
    {
      "@type": "SoftwareApplication",
      name: "Trakt",
      applicationCategory: "BusinessApplication",
      operatingSystem: "Web",
      url: publicConfig.siteUrl,
      description: DESCRIPTION,
      publisher: { "@id": `${publicConfig.siteUrl}/#organization` },
    },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-GB">
      <body>
        {children}
        <script
          type="application/ld+json"
          // Serialised from a literal above; contains no user input.
          dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
        />
      </body>
    </html>
  );
}
