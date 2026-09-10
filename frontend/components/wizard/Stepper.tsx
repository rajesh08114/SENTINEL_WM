import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export function Stepper({ steps, active }: { steps: string[]; active: number }) {
  return (
    <ol className="flex flex-wrap gap-2 font-mono text-xs">
      {steps.map((s, i) => (
        <li key={s} className="flex items-center gap-2">
          <span
            className={cn(
              "grid h-6 w-6 place-items-center rounded-full border",
              i < active
                ? "border-brand bg-brand text-black"
                : i === active
                  ? "border-brand text-brand"
                  : "border-line text-muted"
            )}
          >
            {i < active ? <Check size={12} /> : i + 1}
          </span>
          <span className={i === active ? "text-ink" : "text-muted"}>{s}</span>
          {i < steps.length - 1 && (
            <span className="text-muted" aria-hidden>
              →
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}
