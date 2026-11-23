import type { ReactNode } from "react";

type TagProps = {
  children: ReactNode;
  tone?: "default" | "good";
};

export default function Tag({ children, tone = "default" }: TagProps) {
  const toneClass =
    tone === "good"
      ? "text-emerald-300 border-emerald-500/40"
      : "text-slate-300 border-slate-700";

  return (
    <span className={`inline-block px-2 py-0.5 font-mono text-xs border ${toneClass}`}>
      {children}
    </span>
  );
}
