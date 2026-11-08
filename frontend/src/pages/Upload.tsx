import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud } from "lucide-react";
import { api } from "../lib/api";

const ACCEPT = ".jpg,.jpeg,.png,.mp4,.mov";

type OcrDiagnostics = {
  available: boolean;
  languages: string;
  providers: Record<string, { available: boolean; reason: string | null }>;
};

export function Upload() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [ocrDiagnostics, setOcrDiagnostics] = useState<OcrDiagnostics | null>(null);
  const [diagnosticsError, setDiagnosticsError] = useState("");

  useEffect(() => {
    api.get("/system/media-status")
      .then((response) => setOcrDiagnostics(response.data.ocr))
      .catch(() => setDiagnosticsError("Could not check OCR availability. Processing may fail if no OCR provider is configured."));
  }, []);

  function choose(next: File | null) {
    setFile(next);
    setError("");
    if (preview) URL.revokeObjectURL(preview);
    setPreview(next ? URL.createObjectURL(next) : null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await api.post("/advertisements/upload", form, {
        onUploadProgress: (evt) => {
          if (evt.total) setProgress(Math.round((evt.loaded / evt.total) * 100));
        },
      });
      await api.post(`/advertisements/${res.data.id}/analyze`);
      navigate(`/app/ads/${res.data.id}`);
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || "Upload failed.");
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <h1 className="text-2xl font-semibold">Upload advertisement</h1>
      {diagnosticsError && <p role="alert" className="rounded-xl border border-amber-400/30 bg-amber-400/10 p-4 text-sm text-amber-100">{diagnosticsError}</p>}
      {ocrDiagnostics && !ocrDiagnostics.available && (
        <div role="alert" className="rounded-xl border border-rose-400/30 bg-rose-400/10 p-4 text-sm text-rose-100">
          <p className="font-semibold">OCR is currently unavailable. Uploaded analysis will fail at the OCR stage.</p>
          <ul className="mt-2 list-inside list-disc space-y-1">
            {Object.entries(ocrDiagnostics.providers).map(([provider, status]) =>
              !status.available && status.reason ? <li key={provider}>{provider}: {status.reason}</li> : null
            )}
          </ul>
          <p className="mt-2">Local OCR requires Tesseract with {ocrDiagnostics.languages} language data. Google Cloud Vision is optional.</p>
        </div>
      )}
      <form onSubmit={onSubmit} className="card space-y-5 p-6">
        <label
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            choose(e.dataTransfer.files[0] ?? null);
          }}
          className="flex cursor-pointer flex-col items-center rounded-2xl border border-dashed border-teal-500/40 bg-navy-950/60 px-6 py-12 text-center"
        >
          <UploadCloud className="h-8 w-8 text-teal-400" />
          <p className="mt-3 font-medium">Drag and drop, or choose a file</p>
          <p className="mt-1 text-sm text-slate-400">JPG, JPEG, PNG, MP4, MOV · size limit from server config</p>
          <input
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => choose(e.target.files?.[0] ?? null)}
          />
        </label>
        {file && (
          <div className="text-sm text-slate-300">
            {file.name} · {(file.size / (1024 * 1024)).toFixed(2)} MB
          </div>
        )}
        {preview && file?.type.startsWith("image/") && (
          <img src={preview} alt="Advertisement preview" className="max-h-80 rounded-xl object-contain" />
        )}
        {preview && file?.type.startsWith("video/") && (
          <video src={preview} controls className="max-h-80 w-full rounded-xl" />
        )}
        {progress > 0 && (
          <div className="h-2 overflow-hidden rounded-full bg-white/10">
            <div className="h-full bg-teal-500" style={{ width: `${progress}%` }} />
          </div>
        )}
        {error && <p className="text-sm text-rose-300">{error}</p>}
        <button
          disabled={!file}
          className="rounded-xl bg-teal-600 px-4 py-2.5 text-sm font-semibold disabled:opacity-40"
        >
          Analyze Advertisement
        </button>
      </form>
    </div>
  );
}
