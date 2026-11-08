import { Navigate, Route, Routes } from "react-router-dom";
import { Protected } from "./components/Protected";
import { AppShell } from "./layouts/AppShell";
import { AnalysisStatus } from "./pages/AnalysisStatus";
import { Dashboard } from "./pages/Dashboard";
import { FutureModel } from "./pages/FutureModel";
import { Landing } from "./pages/Landing";
import { Login } from "./pages/Login";
import { Register } from "./pages/Register";
import { Report } from "./pages/Report";
import { Upload } from "./pages/Upload";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route
        path="/app"
        element={
          <Protected>
            <AppShell />
          </Protected>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="upload" element={<Upload />} />
        <Route path="ads/:id" element={<AnalysisStatus />} />
        <Route path="ads/:id/report" element={<Report />} />
        <Route path="future-model" element={<FutureModel />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
