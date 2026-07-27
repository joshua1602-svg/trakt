import { ImageResponse } from "next/og";

/**
 * The social preview image, generated at build time from the same brand tokens
 * as the page — so there is no binary asset to keep in sync, and no
 * placeholder to forget to replace.
 */
export const alt = "Trakt | Governed Portfolio Intelligence";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: "linear-gradient(135deg, #0c1024 0%, #161c38 62%, #232d55 100%)",
          padding: 72,
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 10,
              background: "#232d55",
              border: "1px solid #3d4a82",
            }}
          />
          <div style={{ color: "#eef1f8", fontSize: 32, fontWeight: 600, letterSpacing: -0.5 }}>
            trakt
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              color: "#eef1f8",
              fontSize: 68,
              fontWeight: 600,
              lineHeight: 1.06,
              letterSpacing: -1.8,
            }}
          >
            Portfolio intelligence.
          </div>
          <div
            style={{
              color: "#919dd1",
              fontSize: 68,
              fontWeight: 600,
              lineHeight: 1.06,
              letterSpacing: -1.8,
            }}
          >
            Wherever you work.
          </div>
          <div style={{ color: "#b8c0d6", fontSize: 26, marginTop: 26, maxWidth: 880 }}>
            Trusted answers, automated reporting and governed workflows for specialist
            lending.
          </div>
        </div>

        <div style={{ display: "flex", gap: 14 }}>
          {["Microsoft 365 Copilot", "Teams", "Trakt Workspace", "Automated delivery"].map(
            (channel) => (
              <div
                key={channel}
                style={{
                  color: "#aab4dd",
                  fontSize: 20,
                  border: "1px solid #232b48",
                  borderRadius: 999,
                  padding: "8px 20px",
                }}
              >
                {channel}
              </div>
            ),
          )}
        </div>
      </div>
    ),
    size,
  );
}
