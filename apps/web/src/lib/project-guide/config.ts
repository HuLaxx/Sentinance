/**
 * Per-project settings for the project assistant. Everything else under
 * src/lib/project-guide and src/components/assistant is shared verbatim across
 * FleetFlow, GraphGuard and Sentinance.
 */
export const ASSISTANT_CONFIG = {
  homeProject: "sentinance",
  projectName: "Sentinance",
  ownerName: "Rahul Khanke",
  contactLine: "email rahulkhanke786@gmail.com or message him at linkedin.com/in/rahulkhanke.",
  theme: "sentinance",
  launcherLabel: "Ask Sentinance",
  eyebrow: "Signal desk · RAG guide",
  title: "Ask the Sentinance guide",
  subtitle: "Ask about the project, its stack and evidence, or about Rahul.",
  greeting:
    "Hi! I answer from Sentinance's docs, ADRs and evidence register, and from Rahul's profile. Every answer cites its source.",
  placeholder: "e.g. How do prices reach the dashboard?",
  suggestions: [
    "What is Sentinance and what does it do?",
    "Why LangGraph for the research agent?",
    "How do prices flow from Kafka to PostgreSQL?",
    "What is Rahul's experience and background?",
  ],
  /** Routes where the floating launcher would cover task UI. */
  hiddenPaths: [] as string[],
} as const;
