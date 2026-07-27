import type { MetadataRoute } from "next";

import { publicConfig } from "@/lib/public-config";

/**
 * One indexable document. The in-page anchors are listed for completeness but
 * are not separate URLs, so only the root is emitted.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: `${publicConfig.siteUrl}/`,
      changeFrequency: "monthly",
      priority: 1,
    },
  ];
}
