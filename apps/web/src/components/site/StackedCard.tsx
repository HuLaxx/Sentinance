import type { ReactNode } from "react";

type StackedCardProps = {
  children: ReactNode;
  className?: string;
  innerClassName?: string;
};

export default function StackedCard({
  children,
  className = "",
  innerClassName = "p-7 md:p-8",
}: StackedCardProps) {
  return (
    <div className={`relative ${className}`}>
      <div className="pointer-events-none absolute inset-0 translate-x-3 translate-y-3 rounded-[var(--radius)] border border-white/5 bg-white/[0.02]" />
      <div className="pointer-events-none absolute inset-0 translate-x-1.5 translate-y-1.5 rounded-[var(--radius)] border border-white/10 bg-white/[0.03]" />
      <div className={`relative glass rounded-[var(--radius)] ${innerClassName}`}>{children}</div>
    </div>
  );
}
