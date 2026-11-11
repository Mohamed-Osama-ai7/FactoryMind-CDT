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

  it('renders WP2 regime separation, integrity gating ablations, and simulated fault separation', () => {
    const wp2Model: ModelInfo = {
      ...sampleModel,
      wp2: {
        primary_detector: 'duty_rule',
        warning_detector: 'pca_spe',
        regime_distribution: {
          evaluation: {
            valid_windows: 100,
            invalid_gapped_windows: 2,
            counts: {
              loaded: 10,
              unloaded: 40,
              starting: 5,
              stopped: 25,
              insufficient_evidence: 20,
            },
            insufficient_reasons: { mixed_duty_cycle: 18, stale_telemetry: 2 },
          },
        },
        event_separation: {
          baseline_leak_episodes_wp0: 8,
          baseline_false_alarm_episodes_wp0: 4,
          gated_leak_episodes_wp2: 7,
          gated_false_alarm_episodes_wp2: 3,
          integrity_suppressed_leak_episodes: 1,
          events_detected_wp2: 4,
          events_total: 4,
          false_alarms_per_day_wp2: 0.025,
          operating_false_alarms_per_day_wp2: 0.034,
          sensor_consistency_warning_episodes: 3,
          warnings_per_day: 0.025,
          operating_warnings_per_day: 0.034,
          warning_only_episodes: 2,
          warning_overlapping_leak_episodes: 1,
          warning_kinds: { stale_telemetry: 1, multivariate_sensor_drift: 2 },
          false_alarms_by_dominant_regime: {
            loaded: 2,
            unloaded: 0,
            starting: 1,
            stopped: 0,
            insufficient_evidence: 0,
          },
          starting_overlapping_false_alarms: 1,
          mixed_cycle_overlapping_false_alarms: 1,
        },
        per_regime_metrics: {
          loaded: {
            windows: 10,
            share: 0.1,
            primary: {
              scored_windows: 10,
              exceedance_windows: 5,
              exceedance_rate: 0.5,
              false_alarm_episodes: 2,
              overlapping_false_alarm_episodes: 2,
            },
            warning: {
              scored_windows: 10,
              exceedance_windows: 1,
              exceedance_rate: 0.1,
              warning_episodes: 1,
            },
          },
          stopped: {
            windows: 25,
            share: 0.25,
            primary: {
              scored_windows: 0,
              exceedance_windows: 0,
              exceedance_rate: null,
              false_alarm_episodes: 0,
              overlapping_false_alarm_episodes: 0,
            },
            warning: {
              scored_windows: 0,
              exceedance_windows: 0,
              exceedance_rate: null,
              warning_episodes: 0,
            },
          },
        },
        regime_gating_ablations: {
          wp0_stopped_only: {
            ...sampleModel.loeo,
            suppressed_states: ['stopped'],
            note: 'Baseline stopped suppression.',
          },
          wp2_integrity_gated: {
            events_detected: 4,
            events_total: 4,
            mean_false_alarms_per_day: 0.026,
            mean_operating_false_alarms_per_day: 0.035,
            median_abs_delay_minutes: 50,
            folds: sampleModel.loeo.folds,
            suppressed_states: ['stopped', 'stale_telemetry', 'signal_inconsistency'],
            note: 'Abstains on frozen telemetry.',
          },
          starting_suppressed_ablation: {
            events_detected: 4,
            events_total: 4,
            mean_false_alarms_per_day: 0.026,
            mean_operating_false_alarms_per_day: 0.045,
            median_abs_delay_minutes: 50,
            folds: sampleModel.loeo.folds,
            suppressed_states: ['stopped', 'stale_telemetry', 'signal_inconsistency', 'starting'],
            note: 'Did not improve operating FA/day.',
          },
        },
        simulated_fault_separation: {
          simulated: true,
          real_operational_data: false,
          note: 'Simulated single-sensor faults evaluate warning-tier separation.',
          detectors: {
            duty_rule: {
              detected: 6,
              cases: 60,
              control_detected: 6,
              clean_control_sensitivity: 0.0,
              incremental_lift: 0.0,
              per_type: { stuck: '1/12' },
            },
            pca_spe: {
              detected: 35,
              cases: 60,
              control_detected: 6,
              clean_control_sensitivity: 0.537,
              incremental_lift: 0.483,
              per_type: { stuck: '9/12' },
            },
          },
        },
      },
    }
    render(<ModelCard model={wp2Model} />)
    expect(screen.getByText(/Operating-regime & warning separation \(WP2\)/)).toBeInTheDocument()
    expect(screen.getByText(/1 false leak episode prevented/)).toBeInTheDocument()
    expect(screen.getAllByText('0.026').length).toBeGreaterThanOrEqual(1)
    expect(
      screen.getByText(/pca_spe: 35\/60 gross, ctrl 6\/60, clean sens 53\.7 %, lift \+48\.3 pp/),
    ).toBeInTheDocument()
  })
})
