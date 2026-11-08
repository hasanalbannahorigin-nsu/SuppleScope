import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";

type Stage = { key: string; status: string; detail?: string | null };
type StatusPayload = {
  advertisement_status: string;
  analyze_job: { status: string; error?: string | null };
  research_job: { status: string; error?: string | null };
  stages: Stage[];
};

const LABELS: Record<string, string> = {
  advertisement_uploaded: "Advertisement uploaded",
  video_preprocessing: "Video preprocessing",
  frames_extracted: "Important frames extracted",
  bangla_ocr_completed: "Bangla OCR",
  speech_transcription_completed: "Speech transcription completed",
  information_extraction_completed: "Information extraction completed",
  product_identification_started: "Product identification started",
  ingredient_discovery_started: "Ingredient discovery started",
  scientific_research_started: "Scientific research started",
  safety_research_completed: "Safety research completed",
  research_dossier_generated: "Research dossier generated",
};

export function AnalysisStatus() {
  const { id } = useParams();
  const [data, setData] = useState<StatusPayload | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let timer: number | undefined;
    let cancelled = false;
    async function poll() {
      try {
        const res = await api.get(`/advertisements/${id}/status`);
        if (cancelled) return;
        setData(res.data);
        setError("");
        const busy = ["queued", "running"].includes(res.data.analyze_job.status) ||
          ["queued", "running"].includes(res.data.research_job.status);
        if (busy) timer = window.setTimeout(poll, 2500);
      } catch {
        if (!cancelled) setError("Could not retrieve processing status. Check the backend connection and retry.");
      }
    }
    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [id]);

  async function startResearch() {
    try {
      await api.post(`/advertisements/${id}/research`);
      const res = await api.get(`/advertisements/${id}/status`);
      setData(res.data);
      setError("");
    } catch {
      setError("Could not start research. Check the backend response and retry.");
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <h1 className="text-2xl font-semibold">Analysis status</h1>
      <p className="text-sm text-slate-400">
        Advertisement status: {data?.advertisement_status || "loading"} · stages reflect the backend
        only — completed steps are never simulated.
      </p>
      {error && <p role="alert" className="text-sm text-rose-300">{error}</p>}
      <ol className="space-y-2">
        {(data?.stages || []).map((stage) => (
          <li key={stage.key} className="card flex items-start justify-between p-4 text-sm">
            <div>
              <p className="font-medium">{LABELS[stage.key] || stage.key}</p>
              {stage.detail && <p className="mt-1 text-slate-400">{stage.detail}</p>}
            </div>
            <span
              className={`rounded-full px-2 py-0.5 text-xs ${
                stage.status === "completed"
                  ? "bg-teal-500/20 text-teal-200"
                  : stage.status === "failed"
                    ? "bg-rose-500/20 text-rose-200"
                    : stage.status === "in_progress"
                      ? "bg-sky-500/20 text-sky-200"
                      : stage.status === "skipped"
                        ? "bg-white/10 text-slate-400"
                        : "bg-white/5 text-slate-500"
              }`}
            >
              {stage.status}
            </span>
          </li>
        ))}
      </ol>
      {data?.analyze_job.error && <p className="text-sm text-rose-300">{data.analyze_job.error}</p>}
      {data?.research_job.error && <p className="text-sm text-rose-300">{data.research_job.error}</p>}
      <div className="flex flex-wrap gap-3">
        <button onClick={startResearch} className="rounded-xl bg-teal-600 px-4 py-2 text-sm font-semibold">
          Start agentic research
        </button>
        <Link to={`/app/ads/${id}/report`} className="rounded-xl border border-white/15 px-4 py-2 text-sm">
          Open research report
        </Link>
      </div>
    </div>
  );
}
