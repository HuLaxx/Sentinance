type MetricCardProps = {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
};

export default function MetricCard({ label, value, sub, accent }: MetricCardProps) {
  return (
    <div className="glass rounded-2xl px-5 py-4">
      <div className="text-[11px] font-mono uppercase tracking-widest text-slate-500">{label}</div>
      <div className={`mt-2 text-2xl font-mono ${accent ? "text-emerald-300" : "text-white"}`}>{value}</div>
      {sub ? <div className="mt-1 text-xs text-slate-500">{sub}</div> : null}
    </div>
  );
}
