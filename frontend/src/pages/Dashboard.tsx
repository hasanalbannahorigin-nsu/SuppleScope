import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { FileSearch, FlaskConical, Hourglass, Package, ScrollText } from "lucide-react";
import { api } from "../lib/api";

type Summary = {
  total_advertisements: number;
  total_products: number;
  total_claims: number;
  completed_reports: number;
  pending_jobs: number;
  recent: { id: string; original_filename: string; status: string; media_type: string }[];
};

export function Dashboard() {
  const [data, setData] = useState<Summary | null>(null);

  useEffect(() => {
    api.get("/dashboard/summary").then((res) => setData(res.data));
  }, []);

  const cards = [
    { label: "Advertisements analyzed", value: data?.total_advertisements ?? 0, icon: FileSearch },
    { label: "Products identified", value: data?.total_products ?? 0, icon: Package },
    { label: "Claims extracted", value: data?.total_claims ?? 0, icon: ScrollText },
    { label: "Completed reports", value: data?.completed_reports ?? 0, icon: FlaskConical },
    { label: "Pending research jobs", value: data?.pending_jobs ?? 0, icon: Hourglass },
  ];

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Research dashboard</h1>
          <p className="mt-1 text-sm text-slate-400">User-specific advertisement intelligence workspace.</p>
        </div>
        <Link to="/app/upload" className="rounded-xl bg-teal-600 px-4 py-2.5 text-sm font-semibold">
          Quick upload
        </Link>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {cards.map((card) => (
          <div key={card.label} className="card p-4">
            <card.icon className="h-4 w-4 text-teal-400" />
            <p className="mt-3 text-2xl font-semibold">{card.value}</p>
            <p className="text-xs text-slate-400">{card.label}</p>
          </div>
        ))}
      </div>
      <section className="card p-5">
        <h2 className="font-semibold">Recent analysis history</h2>
        <div className="mt-4 divide-y divide-white/10">
          {(data?.recent || []).length === 0 && (
            <p className="py-6 text-sm text-slate-400">No advertisements yet. Upload a Bangla image or video to begin.</p>
          )}
          {(data?.recent || []).map((row) => (
            <Link
              key={row.id}
              to={`/app/ads/${row.id}`}
              className="flex items-center justify-between py-3 text-sm hover:text-teal-300"
            >
              <span>
                {row.original_filename}{" "}
                <span className="text-slate-500">({row.media_type})</span>
              </span>
              <span className="rounded-full bg-white/5 px-2 py-0.5 text-xs">{row.status}</span>
            </Link>
          ))}
        </div>
      </section>
      <Link
        to="/app/future-model"
        className="card flex items-center justify-between p-5 hover:border-teal-500/40"
      >
        <div>
          <p className="font-semibold">Custom AI Verification Model — Coming Soon</p>
          <p className="mt-1 text-sm text-slate-400">Status: NOT INTEGRATED. No predictions are generated.</p>
        </div>
        <FlaskConical className="h-5 w-5 text-slate-500" />
      </Link>
    </div>
  );
}
