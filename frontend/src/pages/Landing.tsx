import { Link } from "react-router-dom";
import {
  BookOpenText,
  Clapperboard,
  Languages,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { Logo } from "../components/Logo";

const features = [
  {
    icon: Languages,
    title: "Bangla-first OCR",
    body: "Local Tesseract Bengali-English OCR, with optional Google Cloud Vision. Mixed script, numerals, and reading order are preserved — never translated away.",
  },
  {
    icon: Clapperboard,
    title: "Automatic video intelligence",
    body: "FFmpeg and OpenCV extract informative frames, drop near-duplicates, and run OCR on each selected frame. Speech is transcribed separately.",
  },
  {
    icon: Search,
    title: "Agentic product research",
    body: "An API-based research agent searches identity, composition, PubMed, Europe PMC, Crossref, ClinicalTrials.gov, and OpenFDA. It discovers; it does not verdict.",
  },
  {
    icon: BookOpenText,
    title: "Scientific evidence dossier",
    body: "Product-level and ingredient-level literature stay separate. Every finding keeps a source. Uncertainties are shown, never hidden.",
  },
];

export function Landing() {
  return (
    <div>
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <div className="flex items-center gap-3">
          <Logo />
          <div>
            <p className="font-semibold">SuppleScope</p>
            <p className="text-xs text-slate-400">CSE499 senior design research platform</p>
          </div>
        </div>
        <div className="flex gap-3">
          <Link to="/login" className="rounded-xl px-4 py-2 text-sm text-slate-200 hover:bg-white/5">
            Log in
          </Link>
          <Link
            to="/register"
            className="rounded-xl bg-teal-600 px-4 py-2 text-sm font-semibold text-white hover:bg-teal-500"
          >
            Create account
          </Link>
        </div>
      </header>

      <section className="mx-auto grid max-w-6xl gap-10 px-6 py-16 lg:grid-cols-2 lg:items-center">
        <div>
          <p className="text-sm uppercase tracking-[0.2em] text-teal-400">Bangla supplement intelligence</p>
          <h1 className="mt-4 text-4xl font-semibold leading-tight text-white md:text-5xl">
            Look Beyond the Label.
            <span className="block text-teal-300">Discover the Evidence.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg text-slate-300">
            Analyze Bangla supplement advertisements, uncover product ingredients, and explore
            scientific evidence through intelligent research.
          </p>
          <p className="bn mt-3 text-slate-400">বাংলা বিজ্ঞাপন থেকে দাবি, উপাদান ও বৈজ্ঞানিক প্রমাণ খুঁজে বের করুন।</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              to="/register"
              className="rounded-xl bg-teal-600 px-5 py-3 text-sm font-semibold text-white hover:bg-teal-500"
            >
              Start an analysis
            </Link>
            <a href="#how" className="rounded-xl border border-white/15 px-5 py-3 text-sm text-slate-200">
              How it works
            </a>
          </div>
        </div>
        <div className="card p-6">
          <div className="flex items-center gap-2 text-teal-300">
            <Sparkles className="h-4 w-4" /> Research pipeline
          </div>
          <ol className="mt-4 space-y-3 text-sm text-slate-300">
            {[
              "Upload a Bangla advertisement image or short video",
              "Automatic frames, OCR, and speech transcription",
              "Extract product names and original-language claims",
              "Agentic search for composition not shown in the ad",
              "Retrieve real papers, interactions, and disease-risk literature",
              "Review a structured dossier with sources and uncertainty",
            ].map((step, i) => (
              <li key={step} className="flex gap-3">
                <span className="mt-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-teal-600/20 text-xs text-teal-300">
                  {i + 1}
                </span>
                {step}
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-10">
        <h2 className="text-2xl font-semibold">Built for Bangladesh supplement ads</h2>
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {features.map((feature) => (
            <article key={feature.title} className="card p-5">
              <feature.icon className="h-5 w-5 text-teal-400" />
              <h3 className="mt-3 font-semibold">{feature.title}</h3>
              <p className="mt-2 text-sm text-slate-400">{feature.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="how" className="mx-auto max-w-6xl px-6 py-10">
        <h2 className="text-2xl font-semibold">How it works</h2>
        <p className="mt-3 max-w-3xl text-slate-400">
          Users upload JPG, PNG, MP4, or MOV advertisements. The backend stores files in object
          storage, then runs a real job pipeline: preprocessing, scene-aware frame selection, Bangla
          OCR, speech-to-text, information extraction, and optional patient-context-guided safety
          research. Status pages only show stages the backend actually completed.
        </p>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-10">
        <div className="card flex flex-col gap-4 p-6 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold">Custom AI Verification Model</h2>
            <p className="mt-2 text-sm text-slate-400">
              Coming soon — planned for future development. The current agent collects a research
              dossier. It does not produce claim-verification predictions.
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm text-slate-300">
            <ShieldCheck className="h-4 w-4 text-slate-500" /> NOT INTEGRATED
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-10">
        <h2 className="text-2xl font-semibold">About the project</h2>
        <p className="mt-3 max-w-3xl text-slate-400">
          SuppleScope is a university senior design research project (CSE499A/CSE499B): a
          Bangla-first platform for supplement advertisement analysis, product intelligence, and
          biomedical evidence retrieval. It is not a clinical device and does not replace a
          qualified healthcare professional.
        </p>
      </section>

      <footer className="border-t border-white/10 px-6 py-8 text-center text-sm text-slate-500">
        SuppleScope · Look Beyond the Label. Discover the Evidence.
      </footer>
    </div>
  );
}
