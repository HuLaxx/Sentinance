import type { ReactNode } from "react";

type SectionHeaderProps = {
  eyebrow?: string;
  title: string;
  lead?: ReactNode;
  align?: "left" | "center";
  size?: "page" | "section";
};

export default function SectionHeader({
  eyebrow,
  title,
  lead,
  align = "left",
  size = "section",
}: SectionHeaderProps) {
  const alignClasses = align === "center" ? "mx-auto text-center" : "";
  const titleClasses = size === "page" ? "text-4xl md:text-5xl" : "text-3xl md:text-4xl";
  const leadClasses = size === "page" ? "text-lg md:text-xl" : "text-base md:text-lg";
  const Heading = size === "page" ? "h1" : "h2";

  return (
    <div className={`max-w-3xl ${alignClasses}`}>
      {eyebrow ? (
        <div className="text-[11px] font-mono uppercase tracking-widest text-slate-500">{eyebrow}</div>
      ) : null}
      <Heading className={`mt-3 font-display font-semibold tracking-tight ${titleClasses}`}>{title}</Heading>
      {lead ? <p className={`mt-4 text-slate-400/90 leading-relaxed ${leadClasses}`}>{lead}</p> : null}
    </div>
  );
}
