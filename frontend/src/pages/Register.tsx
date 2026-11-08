import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { isAxiosError } from "axios";
import { Logo } from "../components/Logo";
import { useAuth } from "../lib/auth";

export function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      await register(fullName, email, password);
      navigate("/app");
    } catch (err: unknown) {
      if (isAxiosError<{ detail?: string | { msg?: string }[] }>(err)) {
        const detail = err.response?.data?.detail;
        if (typeof detail === "string") {
          setError(detail);
          return;
        }
        if (Array.isArray(detail)) {
          setError(detail.map((item) => item.msg).filter(Boolean).join(" ") || "Registration failed.");
          return;
        }
      }
      setError("Registration failed.");
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6">
      <Link to="/" className="mb-8 flex items-center gap-3">
        <Logo />
        <span className="font-semibold">SuppleScope</span>
      </Link>
      <form onSubmit={onSubmit} className="card space-y-4 p-6">
        <h1 className="text-xl font-semibold">Create your research workspace</h1>
        <label className="block text-sm">
          Full name
          <input
            className="mt-1 w-full rounded-xl border border-white/10 bg-navy-950 px-3 py-2"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            required
          />
        </label>
        <label className="block text-sm">
          Email
          <input
            className="mt-1 w-full rounded-xl border border-white/10 bg-navy-950 px-3 py-2"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </label>
        <label className="block text-sm">
          Password (min 8 characters)
          <input
            className="mt-1 w-full rounded-xl border border-white/10 bg-navy-950 px-3 py-2"
            type="password"
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        {error && <p className="text-sm text-rose-300">{error}</p>}
        <button className="w-full rounded-xl bg-teal-600 py-2.5 font-semibold">Register</button>
        <p className="text-sm text-slate-400">
          Already registered?{" "}
          <Link to="/login" className="text-teal-300">
            Log in
          </Link>
        </p>
      </form>
    </div>
  );
}
