"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Box, Database, LayoutGrid, Shield, Terminal } from "lucide-react";

const navItems = [
  { href: "/", label: "Home", icon: Box },
  { href: "/architecture", label: "Architecture", icon: LayoutGrid },
  { href: "/techdash", label: "Tech Dash", icon: Activity },
  { href: "/data", label: "Data", icon: Database },
  { href: "/demo", label: "Demo", icon: Terminal },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <div className="fixed top-6 left-0 w-full z-50 px-6 flex justify-center">
      <nav className="w-full max-w-[1100px] rounded-full border border-white/10 bg-black/70 backdrop-blur-xl shadow-2xl px-2 py-2 flex items-center justify-between gap-4">
        <Link
          href="/"
          className="focus-ring flex items-center gap-3 px-4 py-2 rounded-full hover:bg-white/5 transition"
        >
          <span className="h-7 w-7 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow">
            <Shield size={14} className="text-white" fill="currentColor" />
          </span>
          <span className="font-display text-sm font-semibold tracking-wide hidden sm:block">
            Sentinance
          </span>
        </Link>

        <div className="hidden lg:flex items-center gap-1 rounded-full bg-white/[0.03] p-1">
          {navItems.map((item) => {
            const isActive = !item.href.includes("#") && pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={isActive ? "page" : undefined}
                className={
                  "focus-ring flex items-center gap-2 rounded-full px-4 py-2 text-xs font-medium transition " +
                  (isActive
                    ? "bg-white text-black"
                    : "text-slate-400 hover:text-white hover:bg-white/5")
                }
              >
                <Icon size={14} />
                {item.label}
              </Link>
            );
          })}
        </div>

        {/* Header Credit */}
        <div className="flex items-center gap-1.5 ml-4 hidden md:flex">
          <span className="hulax-credit-label text-[0.5rem] leading-tight text-white/60 hover:text-white transition-colors duration-300">
            A Project By
          </span>
          <span className="hulax-credit text-[0.7rem] sm:text-sm text-white/80 transition-all duration-300
            hover:font-black hover:text-transparent hover:bg-clip-text hover:bg-gradient-to-r hover:from-blue-600 hover:via-fuchsia-600 hover:to-amber-500
            hover:drop-shadow-[0_0_14px_rgba(214,182,138,0.45)] cursor-default">
            HuLaX
          </span>
        </div>
      </nav>

      <div className="fixed left-0 right-0 top-[78px] px-6 lg:hidden">
        <div className="mx-auto max-w-[1100px] overflow-x-auto">
          <div className="inline-flex items-center gap-2 rounded-full bg-white/[0.03] border border-white/10 px-2 py-2">
            {navItems.map((item) => {
              const isActive = !item.href.includes("#") && pathname === item.href;
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={isActive ? "page" : undefined}
                  className={
                    "focus-ring flex items-center gap-2 whitespace-nowrap rounded-full px-4 py-2 text-xs font-medium transition " +
                    (isActive
                      ? "bg-white text-black"
                      : "text-slate-400 hover:text-white hover:bg-white/5")
                  }
                >
                  <Icon size={14} />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
