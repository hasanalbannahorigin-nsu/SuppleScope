import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Logo } from "../components/Logo";
import { useAuth } from "../lib/auth";

export function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      await login(email, password);
      navigate("/app");
    } catch {
      setError("Invalid email or password.");
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6">
      <Link to="/" className="mb-8 flex items-center gap-3">
        <Logo />
        <span className="font-semibold">SuppleScope</span>
      </Link>
      <form onSubmit={onSubmit} className="card space-y-4 p-6">
        <h1 className="text-xl font-semibold">Log in</h1>
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
          Password
          <input
            className="mt-1 w-full rounded-xl border border-white/10 bg-navy-950 px-3 py-2"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        {error && <p className="text-sm text-rose-300">{error}</p>}
        <button className="w-full rounded-xl bg-teal-600 py-2.5 font-semibold">Continue</button>
        <p className="text-sm text-slate-400">
          New here?{" "}
          <Link to="/register" className="text-teal-300">
            Create an account
          </Link>
        </p>
      </form>
    </div>
  );
}
