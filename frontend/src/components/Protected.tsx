import { Navigate } from "react-router-dom";
import { useAuth } from "../lib/auth";

export function Protected({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) {
    return <div className="p-10 text-slate-300">Loading workspace…</div>;
  }
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}
