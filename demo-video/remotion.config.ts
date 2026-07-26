/**
 * Remotion configuration.
 *
 * Deliberately conservative: H.264, CRF tuned for presentation and website use,
 * and no network access at render time. The browser executable is resolved by
 * `scripts/render.mjs` (see the note there) rather than being hard-coded, so the
 * project renders on a developer machine and in this environment alike.
 */

import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("jpeg");
Config.setJpegQuality(95);
Config.setCodec("h264");
Config.setPixelFormat("yuv420p");
Config.setCrf(18);
Config.setOverwriteOutput(true);
// The Studio/CLI default. scripts/render.mjs derives its own from the core count
// (leaving one core for ffmpeg); override either with REMOTION_CONCURRENCY.
Config.setConcurrency(2);
// GL backend is left at Chrome's default. Every scene is plain CSS and SVG, so
// forcing the software rasteriser ("swangle") bought nothing and measured six
// times slower on this project. Set REMOTION_GL if a host needs a specific one.
Config.setDelayRenderTimeoutInMilliseconds(60000);
