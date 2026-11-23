import SectionHeader from "../../components/site/SectionHeader";
import StackedCard from "../../components/site/StackedCard";

export default function Page() {
  return (
    <div className="pt-36 pb-16 px-6 max-w-5xl mx-auto space-y-8">
      <SectionHeader
        eyebrow="Privacy"
        title="Privacy policy"
        lead="This portfolio experience does not collect personal data beyond basic usage telemetry."
        size="page"
      />
      <StackedCard>
        <p className="text-sm text-slate-400/90 leading-relaxed">
          Sentinance is a portfolio project. Any analytics collected are used to improve the demo
          experience. For production deployments, provide a dedicated privacy policy aligned with
          regulatory requirements in your region.
        </p>
      </StackedCard>
    </div>
  );
}
