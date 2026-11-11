import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import type { RcaAssessment } from '../../api/generated/model'
import { RcaPanel } from './RcaPanel'

const sampleRca: RcaAssessment = {
  rca_id: 'rca-1',
  package_id: 'pkg-1',
  asset_id: 'metropt3-apu-1',
  incident_id: 'a'.repeat(32),
  event_start: '2020-05-29T23:30:00',
  event_end: '2020-05-30T06:00:00',
  abstained: false,
  abstention_reason: null,
  primary_category: 'air_leakage',
  primary_title: 'Pneumatic air leakage downstream or at APU fittings',
  summary:
    'Top ranked hypothesis: Pneumatic air leakage (support_score=0.67, confidence=moderate).',
  uncertainty_note:
    'Ranked scores index consistency with available telemetry; they are not calibrated probabilities.',
  quality_status: 'sufficient',
  confidence_penalty_factor: 1.0,
  score_meaning_note:
    'support_score is a deterministic physical-evidence consistency index (0..1), NOT a calibrated failure probability.',
  ranked_hypotheses: [
    {
      hypothesis_id: 'hyp:air_leakage',
      category: 'air_leakage',
      title: 'Pneumatic air leakage downstream or at APU fittings',
      rank: 1,
      relative_support: 'primary',
      support_score: 0.67,
      confidence_band: 'moderate',
      physical_mechanism: 'Downstream leak forces high compressor load duty.',
      supporting_evidence: [
        {
          ref_id: 'leak:load_duty',
          origin: 'derived',
          phase: 'peak',
          signal_or_feature: 'load_duty',
          statement: 'Load duty reached 0.850 vs baseline 0.120.',
          observed_value: 0.85,
          unit: 'fraction',
          baseline_value: 0.12,
        },
      ],
      contradicting_evidence: [],
      missing_evidence: [
        {
          ref_id: 'missing:flowmeter',
          origin: 'derived',
          signal_or_feature: 'air_demand_flowmeter',
          statement: 'No independent pneumatic flowmeter exists.',
        },
      ],
      correlation_vs_causation_note:
        'Duty exceedance correlates with leaks but does not prove cause.',
      verification_steps: [
        {
          step_id: 'ver:leak_acoustic',
          priority: 'immediate',
          action: 'Check flexible hoses with ultrasonic leak test.',
          expected_if_confirmed: 'Audible hiss or TP3 drop.',
          expected_if_refuted: 'No external pressure bleed.',
          safety_note: 'Advisory field verification only; never actuate equipment.',
        },
      ],
    },
    {
      hypothesis_id: 'hyp:abnormal_high_demand',
      category: 'abnormal_high_demand',
      title: 'Sustained downstream pneumatic demand (without leak)',
      rank: 2,
      relative_support: 'plausible_alternative',
      support_score: 0.37,
      confidence_band: 'low',
      physical_mechanism: 'Heavy door/brake cycling raises airflow consumption.',
      supporting_evidence: [],
      contradicting_evidence: [],
      missing_evidence: [],
      correlation_vs_causation_note:
        'Heavy operational demand stays an alternative without flowmeter.',
      verification_steps: [],
    },
  ],
  correlation_vs_causation_boundary: [
    'Detector scores reflect distributional departure, not causal proof.',
  ],
  provenance: {
    dataset_id: 'metropt3',
    dataset_source: 'https://archive.ics.uci.edu/dataset/791/metropt+3+dataset',
    dataset_license: 'CC BY 4.0',
    preprocessing_version: 'non_interpolated_parquet_v1',
    sampling_contract: '~0.1 Hz nominal',
    window_length: '10min',
    primary_detector: 'duty_rule',
    primary_model_version: 'v1',
  },
  advisory: 'Advisory root-cause hypotheses for human engineering review.',
}

describe('RcaPanel', () => {
  it('renders ranked hypotheses, origin badges, verification steps, and abstention alert', async () => {
    render(<RcaPanel rca={sampleRca} />)
    expect(screen.getByText('Top rank: air_leakage')).toBeInTheDocument()
    expect(screen.getByText(/support index: 0.67/)).toBeInTheDocument()
    expect(screen.getAllByText('Derived feature').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText(/Check flexible hoses/)).toBeInTheDocument()

    // Collapse rank #2 hypothesis details cleanly
    const toggleButtons = screen.getAllByRole('button', { name: 'Hide details' })
    const secondToggle = toggleButtons[1]
    if (!secondToggle) throw new Error('Expected second hypothesis toggle button')
    await userEvent.click(secondToggle)
    expect(screen.getByRole('button', { name: 'Show details' })).toBeInTheDocument()

    // Abstained view renders abstention banner cleanly
    render(
      <RcaPanel
        rca={{
          ...sampleRca,
          abstained: true,
          abstention_reason: 'Coverage 0.25 below reliability threshold.',
          primary_category: 'insufficient_evidence',
        }}
      />,
    )
    expect(screen.getByText('Abstained: insufficient evidence')).toBeInTheDocument()
    expect(screen.getByText('Coverage 0.25 below reliability threshold.')).toBeInTheDocument()
  })
})
