export function PageHeader({
  eyebrow,
  title,
  lead,
  children,
}: {
  eyebrow?: string;
  title: React.ReactNode;
  lead?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <header className="flex flex-col gap-2">
      {eyebrow && (
        <span className="w-max rounded-full border border-line px-2 py-[2px] font-mono text-[11px] uppercase tracking-wide text-muted">
          {eyebrow}
        </span>
      )}
      <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">{title}</h1>
      {lead && <p className="max-w-3xl text-sm leading-relaxed text-muted">{lead}</p>}
      {children}
    </header>
  );
}
