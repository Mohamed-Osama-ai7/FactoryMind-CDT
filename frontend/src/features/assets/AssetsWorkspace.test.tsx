import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { AssetPassportDetail } from './AssetPassportDetail'
import { AssetsOverview } from './AssetsOverview'

const sampleDataset = {
  dataset_id: 'metropt3',
  title: 'MetroPT-3 Air Production Unit Compressor Telemetry',
  asset_ids: ['metropt3-apu-1'],
  source_url: 'https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip',
  source_page: 'https://archive.ics.uci.edu/dataset/791/metropt+3+dataset',
  license: 'CC BY 4.0',
  archive_sha256: 'aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a',
  raw_preserved: true,
  manifest_present: true,
  telemetry_present: true,
  artifact_status: 'verified',
  provenance_origin: 'observed',
  failure_report_count: 4,
  quality: {
    rows_raw: 1516948,
    rows_cleaned: 1516948,
    start: '2020-02-01 00:00:00',
    end: '2020-09-01 03:59:50',
    duplicate_timestamps: 0,
    null_values: 0,
    out_of_range: {},
    non_binary_digital: {},
    median_period_s: 10.0,
    gaps_over_60s: 331,
    gap_hours: 909.52,
    gap_policy: 'preserve_gaps_uninterpolated',
  },
  quality_checks: [
    {
      check_id: 'manifest_provenance',
      status: 'pass',
      summary: 'Manifest SHA-256 and contract verified (1,516,948 cleaned rows).',
    },
  ],
  lineage: [
    {
      step: 1,
      operation: 'raw_acquisition',
      description: 'Preserve raw archive.',
      preserves_raw_source: true,
    },
  ],
}

describe('AssetsOverview and AssetPassportDetail', () => {
  it('blocks unauthorized users lacking asset:read', async () => {
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
        <AssetsOverview />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/do not hold the/)).toBeInTheDocument()
  })

  it('renders asset overview and dataset foundation records cleanly', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        const s = String(url)
        if (s.startsWith('/api/v1/me')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                actor_id: 'eng-1',
                tenant_id: 'acme',
                roles: ['engineer'],
                permissions: ['asset:read', 'anomaly:read'],
                asset_ids: ['metropt3-apu-1'],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        if (s === '/api/v1/assets') {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                items: [
                  {
                    asset_id: 'metropt3-apu-1',
                    name: 'Porto Metro Train APU Screw Compressor #1',
                    asset_type: 'Air Production Unit (APU) Rotary Screw Compressor',
                    sensor_count: 15,
                    failure_mode_count: 4,
                    connected_dataset_ids: ['metropt3'],
                    active_models: ['duty_rule', 'pca_spe'],
                    telemetry_available: true,
                    runtime_model_version: '20261111-v1',
                    safety_boundary: 'advisory-read-only',
                  },
                ],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        return Promise.resolve(
          new Response(JSON.stringify({ items: [sampleDataset] }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <AssetsOverview renderOpenLink={(id) => <button type="button">Inspect {id}</button>} />
      </QueryClientProvider>,
    )

    expect(await screen.findByText('Porto Metro Train APU Screw Compressor #1')).toBeInTheDocument()
    expect(screen.getByText('preserve_gaps_uninterpolated')).toBeInTheDocument()
    expect(screen.getByText(/Manifest SHA-256 and contract verified/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Inspect metropt3-apu-1' })).toBeInTheDocument()
  })

  it('renders full asset passport detail, sensor filtering, and API 404 error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        const s = String(url)
        if (s.startsWith('/api/v1/me')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                actor_id: 'eng-1',
                tenant_id: 'acme',
                roles: ['engineer'],
                permissions: ['asset:read'],
                asset_ids: ['metropt3-apu-1'],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        if (s === '/api/v1/assets/metropt3-apu-1') {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                runtime_model_version: '20261111-v1',
                datasets: [sampleDataset],
                passport: {
                  asset_id: 'metropt3-apu-1',
                  name: 'Porto Metro Train APU Screw Compressor #1',
                  asset_type: 'Rotary Screw Compressor',
                  system_context: 'Railway pneumatic supply.',
                  site_or_domain: 'Metro do Porto',
                  deployment_scope_note: 'Advisory support only.',
                  nominal_sampling_period_s: 10.0,
                  operating_regimes: ['loaded', 'unloaded'],
                  operating_envelopes: {
                    loaded: '5.5 < Motor_current <= 8.0 A',
                    unloaded: '1.5 < Motor_current <= 5.5 A',
                  },
                  sensors: [
                    {
                      sensor_id: 'TP3',
                      display_name: 'Pneumatic panel pressure (TP3)',
                      kind: 'analog',
                      unit: 'bar',
                      physical_role: 'Main panel pressure.',
                      valid_range: [-1, 15],
                      nominal_envelope: '8.2-10.2 bar',
                      unobservability_note: null,
                    },
                    {
                      sensor_id: 'LPS',
                      display_name: 'Low-pressure emergency switch (LPS)',
                      kind: 'digital',
                      unit: 'state (0/1)',
                      physical_role: 'Emergency pressure < 7 bar.',
                      valid_range: [0, 1],
                      nominal_envelope: '0 nominal',
                      unobservability_note: null,
                    },
                  ],
                  failure_modes: [
                    {
                      mode_id: 'air_leakage',
                      display_name: 'Pneumatic Air Leakage',
                      description: 'Air loss in APU circuit.',
                      primary_signatures: ['Elevated loaded duty'],
                      competing_explanations: ['abnormal_high_demand'],
                      recommended_checks: ['Inspect fittings'],
                    },
                  ],
                  connected_dataset_ids: ['metropt3'],
                  model_applicability: [
                    {
                      model_id: 'cross_asset_transfer',
                      status: 'not_calibrated',
                      scope_note: 'Transfer disabled.',
                      limitations: ['MetroPT-3 only'],
                    },
                  ],
                  unobserved_quantities: ['Downstream air-demand flowmeter'],
                  safety_boundary: 'advisory-read-only',
                },
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Asset not found' }), {
            status: 404,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <AssetPassportDetail assetId="metropt3-apu-1" />
        <AssetPassportDetail assetId="missing-asset" />
      </QueryClientProvider>,
    )

    expect(
      await screen.findByText('Explicitly unobserved quantities & boundaries'),
    ).toBeInTheDocument()
    expect(screen.getByText('Downstream air-demand flowmeter')).toBeInTheDocument()
    expect(screen.getByText('not_calibrated')).toBeInTheDocument()
    expect(screen.getByText('TP3')).toBeInTheDocument()
    expect(screen.getByText('LPS')).toBeInTheDocument()
    expect(screen.getByText('Inspect fittings')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'UCI source' })).toHaveAttribute(
      'href',
      sampleDataset.source_page,
    )

    await userEvent.click(screen.getByRole('button', { name: 'digital' }))
    expect(screen.queryByText('TP3')).toBeNull()
    expect(screen.getByText('LPS')).toBeInTheDocument()

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Could not load asset passport: Asset not found/,
    )
  })

  it('shows error alerts on /me, /assets, or /datasets failures', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        const s = String(url)
        if (s.startsWith('/api/v1/me')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                actor_id: 'eng-1',
                tenant_id: 'acme',
                roles: ['engineer'],
                permissions: ['asset:read'],
                asset_ids: ['metropt3-apu-1'],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Storage backend offline' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <AssetsOverview />
      </QueryClientProvider>,
    )

    const alerts = await screen.findAllByRole('alert')
    expect(alerts.length).toBeGreaterThanOrEqual(2)
    expect(alerts.map((a) => a.textContent).join(' ')).toContain('Storage backend offline')
  })

  it('shows permission check alert when /me fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'OIDC unavailable' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <AssetsOverview />
      </QueryClientProvider>,
    )

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Could not verify asset permissions: OIDC unavailable/,
    )
  })
})
