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
        <h1 className="mt-4 text-2xl font-semibold">Custom AI Verification Model</h1>
        <p className="mt-2 text-teal-300">{data?.headline || "Coming Soon — Planned for Future Development."}</p>
        <p className="mt-4 inline-block rounded-full border border-white/15 px-3 py-1 text-xs">
          Status: {data?.status || "NOT INTEGRATED"}
        </p>
        <p className="mt-4 text-sm text-slate-400">{data?.note}</p>
        <div className="mt-6 grid gap-6 md:grid-cols-2">
          <div>
            <h2 className="font-semibold">Future inputs</h2>
            <ul className="mt-2 list-disc pl-5 text-sm text-slate-300">
              {(data?.inputs || []).map((item: string) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h2 className="font-semibold">Expected future outputs</h2>
            <ul className="mt-2 list-disc pl-5 text-sm text-slate-300">
              {(data?.expected_outputs || []).map((item: string) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
        <p className="mt-6 text-sm text-amber-100">
          This page does not generate claim verification, effectiveness scores, or risk predictions.
        </p>
      </div>
    </div>
  );
}
