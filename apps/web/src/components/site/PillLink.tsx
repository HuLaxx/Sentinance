import Link from "next/link";
import type { ComponentType, ReactNode } from "react";

type IconType = ComponentType<{ size?: number; className?: string }>;

type PillLinkProps = {
  href: string;
  children: ReactNode;
  primary?: boolean;
  icon?: IconType;
  className?: string;
};

export default function PillLink({ href, children, primary, icon: Icon, className = "" }: PillLinkProps) {
  return (
    <Link
      href={href}
      className={
        "focus-ring inline-flex items-center justify-center gap-2 rounded-full px-7 py-3 text-sm font-medium tracking-wide transition-all duration-200 " +
        (primary
          ? "bg-white text-black hover:bg-gray-200"
          : "bg-transparent text-white border border-white/15 hover:border-white/30 hover:bg-white/5") +
        (className ? ` ${className}` : "")
      }
    >
      {children}
      {Icon ? <Icon size={16} className="opacity-90" /> : null}
    </Link>
  );
}
