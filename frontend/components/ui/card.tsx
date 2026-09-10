import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("rounded-lg border border-line bg-surface", className)}
      {...p}
    />
  );
}

export function CardHeader({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("border-b border-line px-4 py-3 text-sm font-semibold", className)}
      {...p}
    />
  );
}

export function CardBody({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex flex-col gap-3 p-4", className)} {...p} />;
}

/** convenience: titled section */
export function Panel({
  title,
  children,
  className,
  right,
}: {
  title?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  right?: React.ReactNode;
}) {
  return (
    <Card className={className}>
      {title != null && (
        <CardHeader className="flex items-center justify-between">
          <span>{title}</span>
          {right}
        </CardHeader>
      )}
      <CardBody>{children}</CardBody>
    </Card>
  );
}
