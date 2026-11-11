import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ModelsWorkspace } from './ModelsWorkspace'

const sampleCatalog = {
  separation_disclaimer:
    'Real MetroPT-3 railway telemetry and UCI AI4I 2020 benchmark are evaluated separately.',
  specialists: [
    {
      model_id: 'duty_rule',
      asset_id: 'metropt3-apu-1',
      dataset_id: 'metropt3',
      domain_kind: 'real_operational_telemetry',
      model_family: 'PhysicsDutyBaseline',
      governance_status: 'active',
      uncertainty_mechanism: 'Smoothed conformal calibration',
      scope_note: 'MetroPT-3 leak detector.',
    },
    {
      model_id: 'ai4i-hgb',
      asset_id: 'ai4i-milling-bench',
      dataset_id: 'ai4i2020',
      domain_kind: 'controlled_synthetic_benchmark',
      model_family: 'HistGradientBoostingClassifier',
      governance_status: 'active',
      uncertainty_mechanism: 'Ledoit-Wolf Mahalanobis OOD',
      scope_note: 'AI4I 2020 failure classifier.',
    },
  ],
  ai4i_benchmark_available: true,
  ai4i_selected_model_id: 'ai4i-hgb',
  ai4i_promoted: true,
}

const sampleBenchmark = {
  dataset_id: 'ai4i2020',
  dataset_scope_note: 'UCI AI4I 2020 controlled benchmark (CC BY 4.0).',
  archive_sha256: 'f601f142',
  rows_total: 10000,
  split_counts: { train: 6000, cal: 2000, test: 2000 },
  leakage_free_features: ['Type', 'Air temperature [K]'],
  excluded_target_and_id_columns: ['UDI', 'Machine failure', 'TWF'],
  models: [
    {
      model_id: 'ai4i-hgb',
      model_family: 'HistGradientBoostingClassifier',
      dataset_id: 'ai4i2020',
      cal_metrics: {
        roc_auc: 0.97,
        pr_auc: 0.8,
        brier_score: 0.0125,
        ece_10_bins: 0.004,
        f1_at_threshold: 0.77,
        precision_at_threshold: 0.83,
        recall_at_threshold: 0.72,
        threshold: 0.425,
        selective_coverage_at_margin: 0.996,
        selective_accuracy_at_margin: 0.986,
      },
      test_metrics: {
        roc_auc: 0.9803,
        pr_auc: 0.7984,
        brier_score: 0.0124,
        ece_10_bins: 0.0054,
        f1_at_threshold: 0.7778,
        precision_at_threshold: 0.8448,
        recall_at_threshold: 0.7206,
        threshold: 0.425,
        selective_coverage_at_margin: 0.9945,
        selective_accuracy_at_margin: 0.9879,
      },
      ood_threshold: 5.9,
    },
  ],
  selected_model_id: 'ai4i-hgb',
  promotion_contract: {
    asset_id: 'ai4i-milling-bench',
    dataset_id: 'ai4i2020',
    candidate_model_id: 'ai4i-hgb',
    promoted: true,
    gates: [
      { gate_id: 'leakage_free_features', passed: true, detail: 'Excluded targets.' },
      { gate_id: 'calibration_quality', passed: true, detail: 'Brier=0.0124.' },
    ],
    summary: 'Passed all promotion gates.',
  },
  generated_at: '2026-11-11T14:00:00',
}

describe('ModelsWorkspace', () => {
  it('blocks unauthorized callers lacking anomaly:read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            actor_id: 'aud-1',
            tenant_id: 'acme',
            roles: ['auditor'],
            permissions: ['incident:read', 'audit:read'],
            asset_ids: ['metropt3-apu-1'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <ModelsWorkspace />
      </QueryClientProvider>,
    )
    expect(await screen.findByText(/do not hold the/)).toBeInTheDocument()
  })

  it('renders specialist catalog, AI4I calibration benchmark, and OOD abstention probe', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string, init?: RequestInit) => {
        const s = String(url)
        if (s.startsWith('/api/v1/me')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                actor_id: 'eng-1',
                tenant_id: 'acme',
                roles: ['engineer'],
                permissions: ['anomaly:read'],
                asset_ids: ['metropt3-apu-1'],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        if (s === '/api/v1/ml/catalog') {
          return Promise.resolve(
            new Response(JSON.stringify(sampleCatalog), {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }),
          )
        }
        if (s === '/api/v1/ml/benchmarks/ai4i2020') {
          return Promise.resolve(
            new Response(JSON.stringify(sampleBenchmark), {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }),
          )
        }
        if (s === '/api/v1/ml/predict' && init?.method === 'POST') {
          const payload = JSON.parse(String(init.body)) as { air_temperature_k: number }
          if (payload.air_temperature_k > 330) {
            return Promise.resolve(
              new Response(
                JSON.stringify({
                  asset_id: 'ai4i-milling-bench',
                  selected_model_id: 'ai4i-hgb',
                  model_family: 'HistGradientBoostingClassifier',
                  model_version: 'v1',
                  abstained: true,
                  abstention_message: 'Insufficient evidence to make a reliable diagnosis',
                  abstention_reasons: ['ood_mahalanobis_exceeded'],
                  prediction_label: null,
                  calibrated_failure_probability: null,
                  predicted_failure_modes: [],
                  uncertainty: {
                    mahalanobis_distance: 21.5,
                    ood_threshold: 5.9,
                    is_ood: true,
                    uncertainty_band: 'abstained',
                    calibrated_probability: null,
                    decision_margin: null,
                  },
                  provenance_origin: 'model_prediction',
                  advisory_only: true,
                }),
                { status: 200, headers: { 'Content-Type': 'application/json' } },
              ),
            )
          }
          return Promise.resolve(
            new Response(
              JSON.stringify({
                asset_id: 'ai4i-milling-bench',
                selected_model_id: 'ai4i-hgb',
                model_family: 'HistGradientBoostingClassifier',
                model_version: 'v1',
                abstained: false,
                abstention_message: null,
                abstention_reasons: [],
                prediction_label: 'nominal',
                calibrated_failure_probability: 0.008,
                predicted_failure_modes: [],
                uncertainty: {
                  mahalanobis_distance: 0.85,
                  ood_threshold: 5.9,
                  is_ood: false,
                  uncertainty_band: 'low',
                  calibrated_probability: 0.008,
                  decision_margin: 0.417,
                },
                provenance_origin: 'model_prediction',
                advisory_only: true,
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        return Promise.resolve(new Response('{}', { status: 404 }))
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <ModelsWorkspace />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/Domain separation guarantee/)).toBeInTheDocument()
    expect(screen.getByText('0.9803')).toBeInTheDocument()
    expect(screen.getByText('Promoted · ai4i-hgb')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'ood_shift' }))
    await userEvent.click(screen.getByRole('button', { name: 'Evaluate routed sample' }))
    expect(
      await screen.findByText('Insufficient evidence to make a reliable diagnosis'),
    ).toBeInTheDocument()
  })
})
