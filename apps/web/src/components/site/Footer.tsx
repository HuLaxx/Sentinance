import Link from "next/link";

const footerCols = [
  {
    title: "Product",
    links: [
      { href: "/", label: "Overview" },
      { href: "/demo", label: "Demo" },
    ],
  },
  {
    title: "Platform",
    links: [
      { href: "/architecture", label: "Architecture" },
      { href: "/techdash", label: "Tech Dash" },
      { href: "/data", label: "Data" },
    ],
  },
];

export default function Footer() {
  return (
    <footer className="relative z-10 border-t border-white/10 mt-16 bg-[#050505]">
      <div className="max-w-7xl mx-auto px-6 py-14 grid grid-cols-1 md:grid-cols-3 gap-10">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-full bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center font-display font-bold">
              S
            </div>
            <div>
              <div className="font-display font-semibold">Sentinance</div>
              <div className="text-xs text-slate-500 font-mono">Copyright 2026</div>
            </div>
          </div>
          <p className="text-sm text-slate-500 leading-relaxed">
            Enterprise market intelligence for crypto operators with auditability, guardrails, and
            low-latency execution workflows.
          </p>
        </div>

        {footerCols.map((col) => (
          <div key={col.title}>
            <h4 className="text-white text-sm font-semibold tracking-wide mb-4">{col.title}</h4>
            <ul className="space-y-3 text-sm text-slate-500">
              {col.links.map((link) => (
                <li key={link.href}>
                  <Link href={link.href} className="hover:text-white transition-colors">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="max-w-7xl mx-auto px-6 pb-10 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-slate-600">
        <span>Built for operators. Calm by default. Fast when needed.</span>

        {/* Footer Credit */}
        <div className="inline-flex items-baseline gap-1">
          <span className="hulax-credit-label text-sm text-white/60 transition-all duration-300 hover:text-white hover:drop-shadow-[0_0_8px_rgba(255,255,255,0.5)]">
            A Project By
          </span>
          <span className="hulax-credit text-base sm:text-lg text-white/80 transition-all duration-300 origin-left
            hover:scale-110 hover:font-black hover:text-transparent hover:bg-clip-text hover:bg-gradient-to-r hover:from-sky-300 hover:via-blue-500 hover:to-cyan-400
            hover:drop-shadow-[0_0_14px_rgba(56,189,248,0.45)] cursor-default">
            HuLaX
          </span>
        </div>
      </div>
    </footer>
  );
}
