import SectionHeader from "../../components/site/SectionHeader";
import StackedCard from "../../components/site/StackedCard";

export default function Page() {
  return (
    <div className="pt-36 pb-16 px-6 max-w-5xl mx-auto space-y-8">
      <SectionHeader
        eyebrow="Terms"
        title="Terms of service"
        lead="This portfolio demo is provided as-is for evaluation and demonstration purposes."
        size="page"
      />
      <StackedCard>
        <p className="text-sm text-slate-400/90 leading-relaxed">
          Sentinance market data and analysis are illustrative and not financial advice. For
          production use, publish formal terms that cover acceptable use, data sources, and
          limitation of liability.
        </p>
      </StackedCard>
    </div>
  );
}
