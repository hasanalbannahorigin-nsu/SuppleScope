export function Logo({ className = "h-9 w-9" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 64 64" fill="none" aria-hidden="true">
      <rect width="64" height="64" rx="16" fill="#0B1F3A" />
      <circle cx="28" cy="26" r="12" stroke="#14B8A6" strokeWidth="4" />
      <path d="M37 35 L51 53" stroke="#38BDF8" strokeWidth="4" strokeLinecap="round" />
      <rect x="21" y="22" width="14" height="8" rx="4" fill="#14B8A6" />
      <circle cx="28" cy="26" r="2.5" fill="#E0F2FE" />
    </svg>
  );
}
