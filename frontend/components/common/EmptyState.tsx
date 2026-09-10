import Link from "next/link";

export function EmptyState({
  icon = "◆",
  title,
  body,
  cta,
}: {
  icon?: string;
  title: string;
  body?: React.ReactNode;
  cta?: { href: string; label: string };
}) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-line bg-surface p-10 text-center text-sm text-muted">
      <div className="text-2xl text-brand">{icon}</div>
      <div className="text-ink">{title}</div>
      {body && <div className="max-w-md">{body}</div>}
      {cta && (
        <Link
          href={cta.href}
          className="mt-1 text-brand underline decoration-dotted"
        >
          {cta.label}
        </Link>
      )}
    </div>
  );
}
