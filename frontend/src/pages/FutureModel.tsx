import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import { api } from "../lib/api";

export function FutureModel() {
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    api.get("/future-model/status").then((r) => setData(r.data));
  }, []);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="card p-6">
        <FlaskConical className="h-8 w-8 text-slate-500" />
        <h1 className="mt-4 text-2xl font-semibold">{data?.title || "Claim Verifier and Patient Risk"}</h1>
        <p className="mt-2 text-teal-300">{data?.headline || "Loading status…"}</p>
        <p className="mt-4 inline-block rounded-full border border-white/15 px-3 py-1 text-xs">
          Status: {(data?.status || "…").replace(/_/g, " ")}
          {data?.verifier_mode ? ` · mode ${data.verifier_mode}` : ""}
        </p>
        {(data?.models_loaded || []).length > 0 && (
          <p className="mt-2 text-xs text-slate-400">Models: {data.models_loaded.join(", ")}</p>
        )}
        <p className="mt-4 text-sm text-slate-400">{data?.note}</p>
        <div className="mt-6 grid gap-6 md:grid-cols-2">
          <div>
            <h2 className="font-semibold">Inputs</h2>
            <ul className="mt-2 list-disc pl-5 text-sm text-slate-300">
              {(data?.inputs || []).map((item: string) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h2 className="font-semibold">Outputs</h2>
            <ul className="mt-2 list-disc pl-5 text-sm text-slate-300">
              {(data?.expected_outputs || []).map((item: string) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
        <p className="mt-6 text-sm text-amber-100">
          Run the verifier and the patient risk check from an advertisement's report page (Claim Verification and
          Patient Risk tabs). Results are decision support, not medical advice.
        </p>
      </div>
    </div>
  );
}
