import { InView } from './in-view';

/*
 * Dumbbell chart of the RAG retrieval evaluation (docs/RAG_RETRIEVAL_EVAL.md):
 * dense BGE-small embeddings in a real Qdrant container against a BM25 keyword
 * baseline, on this repo's evaluation corpus. CI fails below the 0.90 gate.
 */
const rows = [
    { metric: 'Recall@1', bm25: 0.625, dense: 0.917 },
    { metric: 'Recall@3', bm25: 0.896, dense: 0.951 },
    { metric: 'Recall@5', bm25: 0.896, dense: 0.979 },
    { metric: 'MRR@10', bm25: 0.755, dense: 0.979 },
];

// Axis runs from 0.5 to 1.0 so the gaps are readable.
const MIN = 0.5;
const scale = (v: number) => (v - MIN) / (1 - MIN);

export function RetrievalChart() {
    return (
        <InView
            className="sx-dumbbell"
            role="img"
            aria-label="Dense retrieval beats the BM25 baseline on every metric: Recall@1 0.917 vs 0.625, Recall@3 0.951 vs 0.896, Recall@5 0.979 vs 0.896, MRR@10 0.979 vs 0.755. The CI gate is 0.90 for Recall@5 and MRR@10."
        >
            <p className="sx-dumbbell__legend" aria-hidden="true">
                <span>
                    <i style={{ background: '#38bdf8' }} />
                    Dense, BGE-small in Qdrant
                </span>
                <span>
                    <i style={{ border: '2px solid #94a3b8' }} />
                    BM25 keyword baseline
                </span>
            </p>
            <div className="sx-rows" aria-hidden="true">
                <div className="sx-gateline" style={{ ['--sx-gate' as string]: scale(0.9) }}>
                    <span>CI gate 0.90</span>
                </div>
                {rows.map((r, i) => (
                    <div className="sx-row" key={r.metric}>
                        <span>{r.metric}</span>
                        <div
                            className="sx-row__track"
                            style={{ ['--sx-from' as string]: scale(r.bm25), ['--sx-to' as string]: scale(r.dense), ['--sx-i' as string]: i }}
                        >
                            <span className="sx-row__bar" />
                            <span className="sx-dot sx-dot--bm25">
                                <b>{r.bm25.toFixed(3)}</b>
                            </span>
                            <span className="sx-dot sx-dot--dense">
                                <b>{r.dense.toFixed(3)}</b>
                            </span>
                        </div>
                    </div>
                ))}
            </div>
            <div className="sx-axis" aria-hidden="true">
                <span />
                <div>
                    <span>0.50</span>
                    <span>0.75</span>
                    <span>1.00</span>
                </div>
            </div>
            <p className="sx-note">
                24 paraphrased questions over a 36-item synthetic news corpus with deliberate near-misses, run against a real
                Qdrant container. Production embeds with Gemini, which CI can&apos;t call, so these scores describe this corpus
                and model.
            </p>
        </InView>
    );
}
