import { FormEvent, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../lib/api";

const TABS = [
  "Overview",
  "Original Advertisement",
  "OCR Results",
  "Video Frames",
  "Speech Transcript",
  "Extracted Claims",
  "Product Identification",
  "Manufacturer Information",
  "Ingredient Composition",
  "Ingredient Research",
  "Scientific Papers",
  "Clinical Evidence",
  "Product Effectiveness Research",
  "Safety Information",
  "Supplement–Drug Interactions",
  "Supplement–Disease Risks",
  "Missing Information",
  "Source References",
];

type Dossier = Record<string, unknown>;

export function Report() {
  const { id } = useParams();
  const [tab, setTab] = useState(TABS[0]);
  const [ocr, setOcr] = useState<any[]>([]);
  const [frames, setFrames] = useState<any[]>([]);
  const [extraction, setExtraction] = useState<any>(null);
  const [report, setReport] = useState<any>(null);
  const [risks, setRisks] = useState<any>(null);
  const [originalUrl, setOriginalUrl] = useState("");
  const [originalIsVideo, setOriginalIsVideo] = useState(false);
  const [ocrFailure, setOcrFailure] = useState("");
  const [patient, setPatient] = useState({ age: "", diseases: "", medications: "", health_context: "" });
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const objectUrls: string[] = [];

    async function load() {
      try {
        const [mediaResponse, ocrResponse, frameResponse, extractionResponse, statusResponse] = await Promise.all([
          api.get(`/advertisements/${id}/original`, { responseType: "blob" }),
          api.get(`/advertisements/${id}/ocr`),
          api.get(`/advertisements/${id}/frames`),
          api.get(`/advertisements/${id}/extraction`),
          api.get(`/advertisements/${id}/status`),
        ]);
        if (cancelled) return;

        const mediaBlob = mediaResponse.data as Blob;
        const mediaUrl = URL.createObjectURL(mediaBlob);
        objectUrls.push(mediaUrl);
        setOriginalUrl(mediaUrl);
        setOriginalIsVideo(mediaBlob.type.startsWith("video/"));

        const frameRows = await Promise.all(
          frameResponse.data.map(async (frame: any) => {
            const imageResponse = await api.get(frame.url, { responseType: "blob" });
            const imageUrl = URL.createObjectURL(imageResponse.data as Blob);
            objectUrls.push(imageUrl);
            return { ...frame, imageUrl };
          })
        );
        if (cancelled) return;
        setOcr(ocrResponse.data);
        const ocrStage = statusResponse.data.stages.find((stage: { key: string }) => stage.key === "bangla_ocr_completed");
        if (ocrResponse.data.length === 0 && ocrStage?.status === "failed") {
          setOcrFailure(ocrStage.detail || statusResponse.data.analyze_job.error || "OCR could not run.");
        } else {
          setOcrFailure("");
        }
        setFrames(frameRows);
        setExtraction(extractionResponse.data);

        api.get(`/advertisements/${id}/patient-context`).then((r) => {
          if (!cancelled) {
            setPatient({
              age: r.data.age ?? "",
              diseases: r.data.diseases ?? "",
              medications: r.data.medications ?? "",
              health_context: r.data.health_context ?? "",
            });
          }
        });
        api.get(`/advertisements/${id}/research-report`)
          .then((r) => { if (!cancelled) setReport(r.data); })
          .catch(() => { if (!cancelled) setReport(null); });
        api.get(`/advertisements/${id}/risk-research`)
          .then((r) => { if (!cancelled) setRisks(r.data); })
          .catch(() => { if (!cancelled) setRisks(null); });
      } catch {
        if (!cancelled) setError("Could not load advertisement analysis. Please retry.");
      }
    }

    load();
    return () => {
      cancelled = true;
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [id]);

  const dossier: Dossier = report?.dossier || {};

  async function savePatient(event: FormEvent) {
    event.preventDefault();
    await api.post(`/advertisements/${id}/patient-context`, {
      age: patient.age === "" ? null : Number(patient.age),
      diseases: patient.diseases,
      medications: patient.medications,
      health_context: patient.health_context,
    });
  }

  async function downloadPdf() {
    const res = await api.get(`/advertisements/${id}/research-report.pdf`, { responseType: "blob" });
    const url = URL.createObjectURL(res.data);
    const a = document.createElement("a");
    a.href = url;
    a.download = "supplescope-dossier.pdf";
    a.click();
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Research report</h1>
          <p className="mt-1 text-sm text-slate-400">
            Structured dossier with provenance. Custom verification model is not integrated.
          </p>
        </div>
        <button onClick={downloadPdf} className="rounded-xl border border-white/15 px-4 py-2 text-sm">
          Export PDF
        </button>
      </div>
      {error && <p role="alert" className="text-sm text-rose-300">{error}</p>}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {TABS.map((name) => (
          <button
            key={name}
            onClick={() => setTab(name)}
            className={`whitespace-nowrap rounded-full px-3 py-1.5 text-xs ${
              tab === name ? "bg-teal-600 text-white" : "bg-white/5 text-slate-300"
            }`}
          >
            {name}
          </button>
        ))}
      </div>
      <section className="card p-5 text-sm leading-6 text-slate-200">
        {tab === "Overview" && (
          <div className="space-y-3">
            <p>
              Identity: {(dossier.product_identity as any)?.product_name || "Unknown"} (
              {(dossier.product_identity as any)?.identity_status || "n/a"})
            </p>
            <p>{(dossier.medical_disclaimer as string) || ""}</p>
            {!report && <p className="text-amber-200">Research dossier not generated yet.</p>}
          </div>
        )}
        {tab === "Original Advertisement" && (
          <div>
            {originalUrl && (originalIsVideo ? (
              <video className="max-h-96 w-full rounded-xl" controls src={originalUrl} />
            ) : (
              <img src={originalUrl} alt="Original advertisement" className="max-h-[480px] rounded-xl" />
            ))}
            {error && <p className="mt-2 text-slate-400">{error}</p>}
          </div>
        )}
        {tab === "OCR Results" && (
          <div className="space-y-4">
            {ocrFailure && <p role="alert" className="rounded-xl border border-rose-400/30 bg-rose-400/10 p-4 text-rose-100">Bangla OCR failed: {ocrFailure}</p>}
            {ocr.map((row) => (
              <article key={row.id} className="rounded-xl bg-navy-950 p-4">
                <p className="text-xs uppercase text-teal-300">{row.provider} · confidence {row.average_confidence ?? "n/a"}</p>
                <p className="text-xs text-slate-500">
                  Source: {row.frame_id ? `video frame ${row.frame_index} at ${row.timestamp_ms} ms` : "original advertisement"}
                </p>
                <p className="bn mt-2 whitespace-pre-wrap">{row.cleaned_text || row.raw_text || "(empty)"}</p>
              </article>
            ))}
            {ocr.length === 0 && <p>No OCR output stored yet.</p>}
          </div>
        )}
        {tab === "Video Frames" && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {frames.map((frame) => (
              <figure key={frame.id} className="overflow-hidden rounded-xl bg-navy-950">
                <img
                  src={frame.imageUrl}
                  alt={`Frame ${frame.frame_index}`}
                  className="h-40 w-full object-cover"
                />
                <figcaption className="p-2 text-xs text-slate-400">
                  Frame {frame.frame_index} · t={frame.timestamp_ms}ms · {frame.selection_reason}
                </figcaption>
              </figure>
            ))}
            {frames.length === 0 && <p>No video frames. Image advertisements skip this stage.</p>}
          </div>
        )}
        {tab === "Speech Transcript" && (
          <div className="space-y-4">
            {(extraction?.transcripts || []).map((transcript: any, i: number) => (
              <article key={i} className="rounded-xl bg-navy-950 p-4">
                <p className="mb-2 text-xs text-slate-400">
                  {transcript.provider} · {transcript.language || "language not identified"}
                </p>
                {transcript.segments?.length ? transcript.segments.map((segment: any, index: number) => (
                  <p key={index} className="bn">
                    <span className="mr-2 text-xs text-teal-300">
                      {segment.start ?? "?"}–{segment.end ?? "?"}s
                    </span>
                    {segment.text}
                  </p>
                )) : <p className="bn whitespace-pre-wrap">{transcript.full_text}</p>}
              </article>
            ))}
            {!extraction?.transcripts?.length && <p>No transcript stored.</p>}
          </div>
        )}
        {tab === "Extracted Claims" && (
          <ul className="space-y-2">
            {(extraction?.claims || []).map((c: any, i: number) => (
              <li key={i} className="bn rounded-xl bg-navy-950 p-3">
                {c.claim_text_original} <span className="text-xs text-slate-500">({c.source})</span>
              </li>
            ))}
            {(extraction?.claims || []).length === 0 && <p>No claims extracted from advertisement text.</p>}
          </ul>
        )}
        {tab === "Product Identification" && (
          <pre className="overflow-x-auto text-xs">{JSON.stringify(dossier.product_identity || extraction?.products, null, 2)}</pre>
        )}
        {tab === "Manufacturer Information" && (
          <p>
            {(dossier.product_identity as any)?.manufacturer || "Not confirmed from retrieved sources."}
            <br />
            {(dossier.product_identity as any)?.official_website}
          </p>
        )}
        {tab === "Ingredient Composition" && (
          <pre className="overflow-x-auto text-xs">{JSON.stringify(dossier.composition, null, 2)}</pre>
        )}
        {tab === "Ingredient Research" && (
          <pre className="overflow-x-auto text-xs">{JSON.stringify(dossier.ingredient_research, null, 2)}</pre>
        )}
        {tab === "Scientific Papers" && (
          <ul className="space-y-3">
            {((dossier.scientific_evidence as any[]) || []).map((p, i) => (
              <li key={i}>
                <a className="text-teal-300" href={p.url} target="_blank" rel="noreferrer">
                  {p.title}
                </a>
                <p className="text-xs text-slate-400">
                  {p.authors} · {p.publication_year} · {p.doi || "no DOI"} · {p.evidence_scope} · {p.source_api}
                </p>
              </li>
            ))}
          </ul>
        )}
        {tab === "Clinical Evidence" && (
          <ul>
            {((dossier.scientific_evidence as any[]) || [])
              .filter((p) => String(p.study_type || "").toLowerCase().includes("trial") || p.source_api === "clinicaltrials")
              .map((p, i) => (
                <li key={i}>
                  <a className="text-teal-300" href={p.url} target="_blank" rel="noreferrer">
                    {p.title}
                  </a>
                </li>
              ))}
          </ul>
        )}
        {tab === "Product Effectiveness Research" && (
          <pre className="overflow-x-auto text-xs">{JSON.stringify(dossier.effectiveness_research, null, 2)}</pre>
        )}
        {tab === "Safety Information" && (
          <pre className="overflow-x-auto text-xs">{JSON.stringify(dossier.side_effects, null, 2)}</pre>
        )}
        {tab === "Supplement–Drug Interactions" && (
          <div>
            <p className="mb-3 text-amber-200">{risks?.disclaimer}</p>
            <pre className="overflow-x-auto text-xs">{JSON.stringify(risks?.supplement_drug_interactions || dossier.supplement_drug_interactions, null, 2)}</pre>
          </div>
        )}
        {tab === "Supplement–Disease Risks" && (
          <pre className="overflow-x-auto text-xs">{JSON.stringify(risks?.supplement_disease_risks || dossier.supplement_disease_risks, null, 2)}</pre>
        )}
        {tab === "Missing Information" && (
          <ul className="list-disc pl-5">
            {(report?.missing_information || dossier.missing_information || []).map((item: string, i: number) => (
              <li key={i}>{item}</li>
            ))}
            {(dossier.uncertainties as string[] || []).map((item, i) => (
              <li key={`u-${i}`} className="text-amber-100">{item}</li>
            ))}
          </ul>
        )}
        {tab === "Source References" && (
          <ul className="space-y-2">
            {((dossier.sources as any[]) || []).map((s, i) => (
              <li key={i}>
                <a className="text-teal-300" href={s.url} target="_blank" rel="noreferrer">
                  {s.title || s.url}
                </a>{" "}
                <span className="text-xs text-slate-500">{s.source_type}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
      <form onSubmit={savePatient} className="card space-y-3 p-5">
        <h2 className="font-semibold">Optional patient context</h2>
        <p className="text-xs text-slate-400">
          Guides interaction and disease-risk searches only. SuppleScope does not diagnose or prescribe.
        </p>
        <div className="grid gap-3 md:grid-cols-2">
          <input
            className="rounded-xl border border-white/10 bg-navy-950 px-3 py-2"
            placeholder="Age"
            value={patient.age}
            onChange={(e) => setPatient({ ...patient, age: e.target.value })}
          />
          <input
            className="rounded-xl border border-white/10 bg-navy-950 px-3 py-2"
            placeholder="Current medications"
            value={patient.medications}
            onChange={(e) => setPatient({ ...patient, medications: e.target.value })}
          />
          <input
            className="rounded-xl border border-white/10 bg-navy-950 px-3 py-2 md:col-span-2"
            placeholder="Relevant diseases"
            value={patient.diseases}
            onChange={(e) => setPatient({ ...patient, diseases: e.target.value })}
          />
          <textarea
            className="rounded-xl border border-white/10 bg-navy-950 px-3 py-2 md:col-span-2"
            placeholder="Other relevant health context"
            value={patient.health_context}
            onChange={(e) => setPatient({ ...patient, health_context: e.target.value })}
          />
        </div>
        <button className="rounded-xl bg-teal-600 px-4 py-2 text-sm font-semibold">Save patient context</button>
      </form>
    </div>
  );
}
