import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { EventEvidencePackage } from '../../api/generated/model'
import { EventEvidencePackageView } from './EventEvidencePackageView'
import { ObservationForm } from './ObservationForm'

const samplePackage: EventEvidencePackage = {
  package_id: 'pkg-1',
  asset_id: 'metropt3-apu-1',
  incident_id: 'a'.repeat(32),
  event_start: '2020-04-18T00:00:00',
  event_end: '2020-04-18T23:59:00',
  raised_at: '2020-04-18T01:30:00',
  lookback_start: '2020-04-17T20:00:00',
  recovery_end: '2020-04-19T01:59:00',
  provenance: {
    dataset_id: 'metropt3',
    dataset_source: 'https://archive.ics.uci.edu/dataset/791/metropt+3+dataset',
    dataset_license: 'CC BY 4.0',
    archive_sha256: 'abc',
    preprocessing_version: 'non_interpolated_parquet_v1',
    sampling_contract: '~0.1 Hz nominal',
    window_length: '10min',
    primary_detector: 'duty_rule',
    primary_model_version: '20261111-f069042373-duty_rule',
    warning_detector: 'pca_spe',
    warning_model_version: '20261111-f069042373-pca_spe',
  },
  phases: [
    {
      phase: 'pre_event_baseline',
      label: 'Pre-event baseline',
      start: '2020-04-17T20:00:00',
      end: '2020-04-18T00:00:00',
      available: true,
      expected_windows: 24,
      total_windows: 24,
      valid_windows: 24,
      sample_count: 1440,
      dominant_regime: 'insufficient_evidence',
      regime_counts: { insufficient_evidence: 24 },
      primary_peak_score: null,
      primary_min_exceedance: null,
      primary_model_origin: 'model_prediction',
      exceedance_origin: 'statistical_inference',
      warning_peak_score: 42.1,
      warning_min_exceedance: 0.002,
      warning_active_windows: 24,
      observed_signals: [
        {
          signal: 'TP3',
          unit: 'bar',
          origin: 'observed',
          sample_count: 1440,
          min: 8.5,
          mean: 8.5,
          max: 8.5,
          std: 0.0,
          last: 8.5,
          frozen: true,
          provenance: 'dataset:metropt3:processed_parquet',
        },
      ],
      derived_features: [
        {
          name: 'unloaded_tp3_slope',
          unit: 'bar/min',
          origin: 'derived',
          value: null,
          observable: false,
          note: 'No unloaded segment in phase',
          provenance: 'window_features:10min',
        },
      ],
      quality_flags: ['stale_telemetry'],
    },
    {
      phase: 'peak',
      label: 'Peak behaviour',
      start: '2020-04-18T07:10:00',
      end: '2020-04-18T07:20:00',
      available: true,
      expected_windows: 1,
      total_windows: 1,
      valid_windows: 1,
      sample_count: 60,
      dominant_regime: 'loaded',
      regime_counts: { loaded: 1 },
      primary_peak_score: 1.0,
      primary_min_exceedance: 0.001,
      primary_model_origin: 'model_prediction',
      exceedance_origin: 'statistical_inference',
      warning_peak_score: 55.2,
      warning_min_exceedance: 0.001,
      warning_active_windows: 1,
      observed_signals: [
        {
          signal: 'TP3',
          unit: 'bar',
          origin: 'observed',
          sample_count: 60,
          min: 8.8,
          mean: 9.0,
          max: 9.2,
          std: 0.1,
          last: 9.1,
          frozen: false,
          provenance: 'dataset:metropt3:processed_parquet',
        },
        {
          signal: 'Motor_current',
          unit: 'A',
          origin: 'observed',
          sample_count: 60,
          min: 5.8,
          mean: 6.1,
          max: 6.4,
          std: 0.2,
          last: 6.2,
          frozen: false,
          provenance: 'dataset:metropt3:processed_parquet',
        },
      ],
      derived_features: [
        {
          name: 'load_duty',
          unit: 'fraction',
          origin: 'derived',
          value: 1.0,
          observable: true,
          note: 'Share under load',
          provenance: 'window_features:10min',
        },
      ],
      quality_flags: [],
    },
  ],
  evidence_items: [
    {
      group: 'operating_state',
      group_label: 'Operating state',
      feature: 'load_duty',
      value: 1.0,
      unit: 'fraction',
      training_median: 0.08,
      robust_z: 9.2,
      novel: false,
      meaning: 'Compressor stayed loaded continuously.',
      origin: 'statistical_inference',
      phase: 'peak',
      source_id: 'feature_profile:duty_rule',
    },
  ],
  quality: {
    overall_status: 'degraded',
    completeness_score: 0.9917,
    coverage_ratio: 0.9861,
    baseline_available: true,
    recovery_available: false,
    recovery_status: 'unverified',
    expected_event_windows: 144,
    valid_event_windows: 142,
    missing_windows_count: 2,
    missing_samples_count: 120,
    stale_windows_count: 2,
    frozen_sensors: ['TP3'],
    out_of_range_signals: [],
    contradictions: [
      {
        kind: 'sensor_freeze',
        timestamp: '2020-04-18T00:00:00',
        severity: 'critical',
        signals: ['TP3', 'TP2', 'Motor_current'],
        description: 'Simultaneously constant signals at onset.',
        origin: 'derived',
      },
    ],
    alignment_issues: ['Recovery/persistence window missing telemetry'],
    confidence_penalty_factor: 0.69,
    quality_warnings: ['Pre-event baseline was stale_telemetry.'],
    limitations: ['~0.1 Hz cadence'],
  },
  missing_measurements: ['Recovery/persistence tail unavailable'],
  conflicting_measurements: ['Stale telemetry freeze at 2020-04-18T00:00:00'],
  neighboring_events: [
    {
      kind: 'operator_report',
      event_id: 'F1',
      title: 'Operator report F1: Air leak',
      start: '2020-04-18T00:00:00',
      end: '2020-04-18T23:59:00',
      detector_or_source: 'operator_report_weak_label',
      relationship: 'concurrent',
      origin: 'human_observation',
    },
  ],
  human_observations: [],
  timeline: [
    {
      timestamp: '2020-04-18T00:00:00',
      category: 'consistency_warning',
      origin: 'derived',
      phase: 'pre_event_baseline',
      title: 'Consistency warning (stale_telemetry)',
      summary: 'Telemetry remained frozen prior to onset.',
      source_id: 'warning:signal_integrity',
      regime: 'insufficient_evidence',
      event_class: 'sensor_consistency_warning',
      details: {},
    },
    {
      timestamp: '2020-04-18T07:10:00',
      category: 'alarm_episode',
      origin: 'model_prediction',
      phase: 'peak',
      title: 'Air-leak alarm episode (loaded)',
      summary: 'Sustained duty exceedance.',
      source_id: 'model:duty_rule',
      regime: 'loaded',
      event_class: 'air_leak_alarm',
      details: {},
    },
  ],
}

describe('EventEvidencePackageView', () => {
  it('separates observed signals from derived features and lets engineers inspect phases and filter timeline', async () => {
    render(<EventEvidencePackageView pkg={samplePackage} />)
    expect(screen.getByText('Quality status: degraded')).toBeInTheDocument()
    expect(screen.getByText('99.2 %')).toBeInTheDocument()
    expect(screen.getByText('Recovery/persistence tail unavailable')).toBeInTheDocument()
    expect(screen.getByText('~0.1 Hz cadence')).toBeInTheDocument()
    expect(screen.getByText('TP3 (bar)')).toBeInTheDocument()
    expect(screen.getByText('Compressor stayed loaded continuously.')).toBeInTheDocument()
    expect(screen.getByText('Motor_current')).toBeInTheDocument()
    expect(screen.getByText(/1 fraction/)).toBeInTheDocument()

    // Switch phase to Pre-event baseline to view unobservable slope and frozen TP3
    await userEvent.click(screen.getByRole('button', { name: /Pre-event baseline/ }))
    expect(screen.getByText('frozen (const)')).toBeInTheDocument()
    expect(screen.getByText(/Unobservable/)).toBeInTheDocument()

    // Filter timeline to alarm_episode only
    expect(screen.getByText(/Chronological event timeline \(2\)/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'alarm episode' }))
    expect(screen.getByText(/Chronological event timeline \(1\)/)).toBeInTheDocument()
    expect(screen.getByText(/Air-leak alarm episode/)).toBeInTheDocument()
  })

  it('shows ObservationForm only when the principal has observation:create permission', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string, init?: RequestInit) => {
        if (String(url).startsWith('/api/v1/me')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                actor_id: 'eng-1',
                tenant_id: 'acme',
                roles: ['engineer'],
                permissions: ['incident:read', 'observation:create'],
                asset_ids: ['metropt3-apu-1'],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'obs-1',
              incident_id: 'a'.repeat(32),
              asset_id: 'metropt3-apu-1',
              actor_id: 'eng-1',
              category: 'field_inspection',
              observed_at: '2020-04-18T00:00:00',
              note: 'Inspected valve manifold.',
              created_at: '2020-04-18T02:00:00',
              origin: 'human_observation',
            }),
            {
              status: init?.method === 'POST' ? 201 : 200,
              headers: { 'Content-Type': 'application/json' },
            },
          ),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const invSpy = vi.spyOn(client, 'invalidateQueries')
    render(
      <QueryClientProvider client={client}>
        <ObservationForm incidentId={'a'.repeat(32)} defaultTimestamp="2020-04-18T00:00:00" />
      </QueryClientProvider>,
    )

    const noteField = await screen.findByLabelText(/Observation note/)
    await userEvent.type(noteField, 'Inspected valve manifold.')
    await userEvent.click(screen.getByRole('button', { name: 'Record observation' }))
    expect(await screen.findByText(/Observation recorded and audited/)).toBeInTheDocument()
    expect(invSpy).toHaveBeenCalledWith({
      queryKey: [`/api/v1/incidents/${'a'.repeat(32)}/rca`],
    })
  })
})
