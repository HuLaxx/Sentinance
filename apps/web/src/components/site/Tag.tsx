import type { ReactNode } from "react";

type TagProps = {
  children: ReactNode;
  tone?: "default" | "good";
};

export default function Tag({ children, tone = "default" }: TagProps) {
  const toneClass =
    tone === "good"
      ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/25"
      : "bg-white/[0.03] text-slate-400 border-white/10";

  return (
    <span className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest border ${toneClass}`}>
      {children}
    </span>
  );
}
