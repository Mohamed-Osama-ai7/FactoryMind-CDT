import { useId, useState } from 'react'
import type { EvidenceReference, RankedHypothesis, RcaAssessment } from '../../api/generated/model'

export interface RcaPanelProps {
  rca: RcaAssessment
  headingLevel?: 'h2' | 'h3' | 'h4'
}

const ORIGIN_LABELS: Record<string, string> = {
  observed: 'Observed',
  derived: 'Derived feature',
  statistical_inference: 'Statistical inference',
  model_prediction: 'Model prediction',
  human_observation: 'Human observation',
}

const BAND_STYLES: Record<string, string> = {
  high: 'border-ok/60 text-ok',
  moderate: 'border-accent/60 text-accent',
  low: 'border-warn/60 text-warn',
  abstained: 'border-critical/60 text-critical',
}

const SUPPORT_LABELS: Record<string, string> = {
  primary: 'Primary hypothesis',
  plausible_alternative: 'Plausible alternative',
  weak: 'Weak support',
  ruled_out: 'Ruled out',
}

function EvidenceRefList({ title, refs }: { title: string; refs: EvidenceReference[] }) {
  if (refs.length === 0) return null
  return (
    <div>
      <p className="text-xs font-medium text-ink">{title}</p>
      <ul className="mt-1 space-y-1 text-xs text-muted">
        {refs.map((r) => (
          <li key={`${r.ref_id}-${r.signal_or_feature}-${r.statement}`}>
            <span className="rounded border border-border px-1 py-0.5 text-[11px] text-ink">
              {ORIGIN_LABELS[r.origin] ?? r.origin}
            </span>{' '}
            <span className="font-mono text-ink">{r.signal_or_feature}</span>
            {r.phase ? ` (${r.phase})` : ''}: {r.statement}
            {r.observed_value !== null && r.observed_value !== undefined && (
              <span>
                {' '}
                [observed: {r.observed_value.toFixed(3)}
                {r.unit ? ` ${r.unit}` : ''}
                {r.baseline_value !== null && r.baseline_value !== undefined
                  ? ` vs baseline ${r.baseline_value.toFixed(3)}`
                  : ''}
                ]
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

function HypothesisCard({ hyp, defaultOpen }: { hyp: RankedHypothesis; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const bandStyle = BAND_STYLES[hyp.confidence_band] ?? 'border-border text-ink'
  const supporting = hyp.supporting_evidence ?? []
  const contradicting = hyp.contradicting_evidence ?? []
  const missing = hyp.missing_evidence ?? []
  const steps = hyp.verification_steps ?? []

  return (
    <div className="rounded border border-border p-3 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono font-semibold text-ink">#{hyp.rank}</span>
          <span className="font-medium text-ink">{hyp.title}</span>
          <span className="rounded border border-border px-1.5 py-0.5 text-[11px] text-muted">
            {SUPPORT_LABELS[hyp.relative_support] ?? hyp.relative_support}
          </span>
          <span className={`rounded border px-1.5 py-0.5 text-[11px] font-medium ${bandStyle}`}>
            confidence: {hyp.confidence_band}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-muted">
            support index: {hyp.support_score.toFixed(2)}
          </span>
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
            className="text-accent underline"
          >
            {open ? 'Hide details' : 'Show details'}
          </button>
        </div>
      </div>

      {open && (
        <div className="mt-3 flex flex-col gap-3 border-t border-border pt-3">
          <p className="text-muted">
            <strong className="text-ink">Mechanism:</strong> {hyp.physical_mechanism}
          </p>
          <EvidenceRefList title="Supporting evidence" refs={supporting} />
          <EvidenceRefList title="Contradicting evidence" refs={contradicting} />
          <EvidenceRefList title="Missing / unmeasured context" refs={missing} />
          <p className="rounded border border-border/80 bg-surface px-2.5 py-1.5 text-muted">
            <strong className="text-ink">Correlation vs causation:</strong>{' '}
            {hyp.correlation_vs_causation_note}
          </p>
          {steps.length > 0 && (
            <div>
              <p className="font-medium text-ink">Advisory verification steps</p>
              <ul className="mt-1 space-y-1.5 text-muted">
                {steps.map((s) => (
                  <li key={s.step_id} className="rounded border border-border/60 p-2">
                    <span className="font-medium text-ink">
                      [{s.priority}] {s.action}
                    </span>
                    <span className="block">
                      Confirmed: {s.expected_if_confirmed} · Refuted: {s.expected_if_refuted}
                    </span>
                    <span className="block text-[11px]">{s.safety_note}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/** Explainable multi-hypothesis root-cause analysis grounded in an EventEvidencePackage. */
export function RcaPanel({ rca, headingLevel = 'h2' }: RcaPanelProps) {
  const Heading = headingLevel
  const headingId = useId()
  const topBand = rca.ranked_hypotheses[0]?.confidence_band ?? 'low'
  const topStyle = BAND_STYLES[topBand] ?? 'border-border text-ink'

  return (
    <section
      aria-labelledby={headingId}
      className="flex flex-col gap-4 rounded-lg border border-border bg-panel p-4 text-sm"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Heading id={headingId} className="font-medium">
          Explainable root-cause analysis (advisory)
        </Heading>
        <span className={`rounded border px-2 py-0.5 text-xs font-medium ${topStyle}`}>
          {rca.abstained ? 'Abstained: insufficient evidence' : `Top rank: ${rca.primary_category}`}
        </span>
      </div>

      {rca.abstained && rca.abstention_reason && (
        <p
          role="status"
          className="rounded border border-critical/60 bg-critical/10 p-2.5 text-xs text-critical"
        >
          {rca.abstention_reason}
        </p>
      )}

      <div className="text-xs">
        <p className="font-medium text-ink">{rca.summary}</p>
        <p className="mt-1 text-muted">{rca.uncertainty_note}</p>
        {rca.score_meaning_note && <p className="mt-1 text-muted">{rca.score_meaning_note}</p>}
      </div>

      <div className="flex flex-col gap-2">
        {rca.ranked_hypotheses.map((hyp, i) => (
          <HypothesisCard key={hyp.hypothesis_id} hyp={hyp} defaultOpen={i < 2} />
        ))}
      </div>

      <div className="border-t border-border pt-3 text-xs text-muted">
        <p className="font-medium text-ink">Correlation vs causation boundary</p>
        <ul className="mt-1 list-disc pl-5">
          {rca.correlation_vs_causation_boundary.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        {rca.advisory && <p className="mt-2">{rca.advisory}</p>}
      </div>
    </section>
  )
}

