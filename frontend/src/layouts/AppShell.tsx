import { NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  FlaskConical,
  LayoutDashboard,
  LogOut,
  Microscope,
  ShieldAlert,
  Upload,
} from "lucide-react";
import { Logo } from "../components/Logo";
import { useAuth } from "../lib/auth";

const links = [
  { to: "/app", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/app/upload", label: "Upload", icon: Upload, end: false },
  { to: "/app/future-model", label: "Future Model", icon: FlaskConical, end: false },
];

export function AppShell() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
      <aside className="border-b border-white/10 bg-navy-900/90 p-5 lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-3">
          <Logo />
          <div>
            <p className="text-sm font-semibold tracking-wide text-white">SuppleScope</p>
            <p className="text-xs text-teal-400">Evidence platform</p>
          </div>
        </div>
        <nav className="mt-8 space-y-1">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${
                  isActive ? "bg-teal-600/20 text-teal-300" : "text-slate-300 hover:bg-white/5"
                }`
              }
            >
              <link.icon className="h-4 w-4" />
              {link.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-8 rounded-xl border border-amber-400/20 bg-amber-400/5 p-3 text-xs text-amber-100">
          <p className="flex items-center gap-2 font-semibold">
            <ShieldAlert className="h-4 w-4" /> Medical disclaimer
          </p>
          <p className="mt-2 text-amber-100/80">
            Research only. Not a diagnosis, prescription, or verification verdict.
          </p>
        </div>
        <button
          className="mt-6 flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm text-slate-400 hover:bg-white/5"
          onClick={() => {
            logout();
            navigate("/");
          }}
        >
          <LogOut className="h-4 w-4" /> Sign out
        </button>
      </aside>
      <div>
        <header className="flex items-center justify-between border-b border-white/10 px-6 py-4">
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Microscope className="h-4 w-4 text-teal-400" />
            Look beyond the label.
          </div>
          <div className="text-sm text-slate-300">{user?.full_name}</div>
        </header>
        <main className="p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
