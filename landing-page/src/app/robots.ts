import type { MetadataRoute } from "next";

import { publicConfig } from "@/lib/public-config";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        // The demo API is stateful and rate-limited; there is nothing to index.
        disallow: ["/api/"],
      },
    ],
    sitemap: `${publicConfig.siteUrl}/sitemap.xml`,
    host: publicConfig.siteUrl,
  };
}
