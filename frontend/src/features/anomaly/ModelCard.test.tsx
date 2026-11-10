import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ModelInfo } from '../../api/generated/model'
import { ModelCard } from './ModelCard'

const sampleModel: ModelInfo = {
  model_version: '20261110-f069042373-duty_rule',
  detector: 'duty_rule',
  selected_as_default: true,
  selection_rationale: 'no learned detector qualified; duty_rule is the best rule baseline',
  dataset_id: 'metropt3',
  dataset_license: 'CC BY 4.0',
  dataset_source: 'https://archive.ics.uci.edu/dataset/791/metropt+3+dataset',
  data_kind: 'real operational telemetry (public dataset)',
  training_period: ['2020-02-01', '2020-03-17'],
  threshold: 0.5,
  persistence: 6,
  exceedance_at_threshold: 0.1199,
  window_length: '10min',
  loeo: {
    events_detected: 4,
    events_total: 4,
    mean_false_alarms_per_day: 0.032,
    mean_operating_false_alarms_per_day: 0.041,
    median_abs_delay_minutes: 50,
    folds: [{ held_out: 'F1', detected: true, delay_minutes: 90 }],
  },
  fault_injection: {
    detected: 6,
    cases: 60,
    gross_rate: 0.1,
    control_detected: 6,
    control_alarm_rate: 0.1,
    control_alarms: '6/60',
    clean_control_cases: 54,
    clean_control_detected: 0,
    clean_control_sensitivity: 0.0,
    incremental_lift: 0.0,
    per_type: { stuck: '1/12' },
    simulated: true,
  },
  reference_holdout_check: {
    alpha: 0.01,
    share_at_or_below_alpha: 0.275,
    windows: 600,
  },
  exceedance_check: {
    alpha: 0.01,
    share_at_or_below_alpha: 0.0336,
    by_month: { '2020-04': 0.0253 },
    windows: 12789,
  },
  intended_use: 'Advisory early warning.',
  limitations: ['Non-stationary reference.'],
  replay_available_from: '2020-02-01T00:00:00',
  replay_available_to: '2020-09-01T00:00:00',
  operator_reports: [],
}

describe('ModelCard', () => {
  it('displays distinguishable valid-day and operating-day false alarm rates, fault lift metrics, and calibration holdout check', () => {
    render(<ModelCard model={sampleModel} />)
    expect(
      screen.getByText(/0\.032 false alarms \/ valid day \(0\.041 false alarms \/ operating day\)/),
    ).toBeInTheDocument()
    expect(screen.getByText(/gross detection 6\/60 \(10\.0 %\)/)).toBeInTheDocument()
    expect(screen.getByText(/clean-control sensitivity 0\/54 \(0\.0 %\)/)).toBeInTheDocument()
    expect(screen.getByText(/incremental lift \+0\.0 pp/)).toBeInTheDocument()
    expect(screen.getByText(/Calibration reference check/)).toBeInTheDocument()
    expect(screen.getByText('27.5 %')).toBeInTheDocument()
    expect(screen.getByText(/never a failure probability/)).toBeInTheDocument()
  })
})
