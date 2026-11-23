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
    <div className={`atlas-panel ${className}`}>
      <div className={innerClassName}>{children}</div>
    </div>
  );
}
