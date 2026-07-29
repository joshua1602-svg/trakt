import { useState } from "react";
import { FileText, Lock } from "lucide-react";
import type { ConfigLayer, PackageFile } from "@/api/adminTypes";
import { useOpsClient } from "@/api/context";
import { ErrorNote, Loading } from "@/components/ErrorNote";
import { copy } from "@/lib/copy";
import { useLoad } from "@/lib/useLoad";
import { EmptyNote, TechnicalDetails } from "./primitives";

function FileContent({
  layer,
  version,
  file,
}: {
  layer: ConfigLayer;
  version: number;
  file: PackageFile;
}) {
  const client = useOpsClient();
  const { data, error, loading, reload } = useLoad(
    () => client.getConfigVersion(layer, version, file.path),
    [layer, version, file.path],
  );
  if (loading) return <Loading />;
  if (error) return <ErrorNote message={error} onRetry={() => void reload()} />;
  return (
    <pre className="max-h-96 overflow-auto whitespace-pre rounded-xl bg-stone-50 p-4 font-mono text-xs text-stone-700">
      {data?.file_content ?? ""}
    </pre>
  );
}

/**
 * Read-only inspection of a package's parts: name, fingerprint, size and
 * contents. Nothing here can write — active versions are never editable.
 */
export function FileInspector({
  layer,
  version,
  files,
}: {
  layer: ConfigLayer;
  version: number;
  files: PackageFile[];
}) {
  const [openPath, setOpenPath] = useState<string | null>(null);

  if (files.length === 0) {
    return <EmptyNote>{copy.admin.files.empty}</EmptyNote>;
  }

  return (
    <div className="space-y-2">
      <p className="flex items-center gap-2 text-xs text-stone-500">
        <Lock className="h-3.5 w-3.5" aria-hidden />
        {copy.admin.files.readOnly}
      </p>
      {files.map((file) => {
        const open = openPath === file.path;
        return (
          <div key={file.path} className="rounded-xl border border-stone-200 bg-white">
            <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
              <span className="flex min-w-0 items-center gap-3">
                <FileText className="h-4 w-4 shrink-0 text-stone-400" aria-hidden />
                <span className="truncate text-sm font-medium text-stone-900">{file.label}</span>
              </span>
              <span className="flex shrink-0 items-center gap-3 text-xs text-stone-500">
                <span>
                  {copy.admin.files.size}: {file.size}
                </span>
                <button
                  type="button"
                  onClick={() => setOpenPath(open ? null : file.path)}
                  className="rounded-lg border border-stone-300 bg-white px-3 py-1 text-xs font-medium text-stone-700 hover:bg-stone-50"
                >
                  {open ? copy.admin.actions.closePart : copy.admin.actions.viewPart}
                </button>
              </span>
            </div>
            {open && (
              <div className="border-t border-stone-100 px-4 py-3">
                <FileContent layer={layer} version={version} file={file} />
                <TechnicalDetails>
                  <p>{file.path}</p>
                  <p>{file.sha256}</p>
                </TechnicalDetails>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
