import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ReplayResponse } from '../../api/generated/model'
import { ReplayPanel } from './ReplayPanel'

// ECharts needs a real canvas; chart rendering is not under test here.
vi.mock('./ScoreChart', () => ({
  ScoreChart: (props: { windows: unknown[] }) => (
    <div data-testid="score-chart">{props.windows.length} windows</div>
  ),
}))

const replay: ReplayResponse = {
  model_version: 'v-test',
  threshold: 0.7,
  persistence: 2,
  start: '2020-06-05T09:00:00',
  end: '2020-06-05T10:00:00',
  windows: [
    {
      window_start: '2020-06-05T09:00:00',
      window_end: '2020-06-05T09:10:00',
      n_samples: 60,
      status: 'normal',
      score: 0.5,
      exceedance: 0.4,
      evidence: [],
      event_class: 'informational',
      event_meaning: 'Nominal operating telemetry.',
      regime: {
        regime: 'unloaded',
        confidence: 'confirmed',
        reason: 'confirmed_unloaded',
        summary: 'Compressor running unloaded for 100% of the window.',
        quality_flags: [],
        missing_signals: [],
      },
    },
    {
      window_start: '2020-06-05T09:10:00',
      window_end: '2020-06-05T09:20:00',
      n_samples: 12,
      status: 'insufficient_data',
      score: null,
      exceedance: null,
      evidence: [],
      event_class: 'insufficient_evidence',
      event_meaning: 'Telemetry window lacks valid sample coverage.',
      regime: {
        regime: 'insufficient_evidence',
        confidence: 'insufficient_evidence',
        reason: 'incomplete_or_gapped',
        summary: 'Window fails sample coverage requirements.',
        quality_flags: ['incomplete_or_gapped'],
        missing_signals: ['valid_non_stale_telemetry'],
      },
    },
    {
      window_start: '2020-06-05T09:20:00',
      window_end: '2020-06-05T09:30:00',
      n_samples: 60,
      status: 'anomalous',
      score: 0.8,
      exceedance: 0.01,
      evidence: [
        {
          group: 'operating_state',
          group_label: 'Operating state',
          feature: 'load_duty',
          value: 0.9,
          training_median: 0.1,
          robust_z: 12.5,
          novel: false,
          meaning: 'Share of the window the compressor spent under load.',
        },
      ],
      event_class: 'air_leak_alarm',
      event_meaning:
        'Sustained excess compressor load duty consistent with an air leak hypothesis.',
      regime: {
        regime: 'loaded',
        confidence: 'confirmed',
        reason: 'confirmed_loaded',
        summary: 'Compressor under load for 90% of the window.',
        quality_flags: [],
        missing_signals: [],
      },
    },
    {
      window_start: '2020-06-05T09:30:00',
      window_end: '2020-06-05T09:40:00',
      n_samples: 60,
      status: 'anomalous',
      score: 0.75,
      exceedance: 0.02,
      evidence: [
        {
          group: 'pressure_level',
          group_label: 'Pressure level',
          feature: 'TP2_mean',
          value: 9.1,
          training_median: 0.0,
          robust_z: 4.0,
          novel: false,
          meaning: 'Compressor outlet pressure (TP2).',
        },
      ],
      event_class: 'air_leak_alarm',
      event_meaning:
        'Sustained excess compressor load duty consistent with an air leak hypothesis.',
      regime: {
        regime: 'loaded',
        confidence: 'confirmed',
        reason: 'confirmed_loaded',
        summary: 'Compressor under load for 90% of the window.',
        quality_flags: [],
        missing_signals: [],
      },
    },
  ],
  episodes: [
    {
      start: '2020-06-05T09:20:00',
      end: '2020-06-05T09:40:00',
      raised_at: '2020-06-05T09:40:00',
      peak_score: 0.8,
      truncated: false,
      ongoing: false,
    },
  ],
  operator_reports: [
    {
      report_id: 'F3',
      start: '2020-06-05T10:00:00',
      end: '2020-06-07T14:30:00',
      failure: 'Air leak',
    },
  ],
}

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <ReplayPanel start="2020-06-05T09:00:00" hours={6} />
    </QueryClientProvider>,
  )
}

const VIEWER = {
  actor_id: 'vera',
  tenant_id: 'acme',
  roles: ['viewer'],
  permissions: ['anomaly:read', 'incident:read'],
  asset_ids: ['metropt3-apu-1'],
}

function respond(body: unknown, status = 200, me: unknown = VIEWER) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((url: string) => {
      const isMe = String(url).startsWith('/api/v1/me')
      return Promise.resolve(
        new Response(JSON.stringify(isMe ? me : body), {
          status: isMe ? 200 : status,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
    }),
  )
}

describe('ReplayPanel', () => {
  it('requests the computed range and summarises windows, episodes and reports', async () => {
    respond(replay)
    renderPanel()
    expect(await screen.findByTestId('score-chart')).toHaveTextContent('4 windows')
    expect(
      screen.getByText(
        /1 normal, 2 above threshold, 0 machine stopped \(explained\), 1 with insufficient\s+data/,
      ),
    ).toBeInTheDocument()
    expect(screen.getByText(/F3 Air leak/)).toBeInTheDocument()
    const url = String(
      vi.mocked(fetch).mock.calls.find(([u]) => String(u).includes('/anomaly/replay'))?.[0],
    )
    expect(url).toContain('start=2020-06-05T09%3A00%3A00')
    expect(url).toContain('end=2020-06-05T15%3A00%3A00')
  })

  it('shows evidence from the peak window of a selected episode', async () => {
    respond(replay)
    renderPanel()
    await userEvent.click(await screen.findByRole('button', { name: 'Show evidence' }))
    expect(screen.getByText('load_duty')).toBeInTheDocument()
    expect(screen.getByText('12.5')).toBeInTheDocument()
    expect(screen.queryByText('TP2_mean')).not.toBeInTheDocument()
  })

  it('shows an empty state when no telemetry exists in range', async () => {
    respond({ ...replay, windows: [], episodes: [], operator_reports: [] })
    renderPanel()
    expect(await screen.findByText(/No telemetry recorded/)).toBeInTheDocument()
  })

  it('surfaces backend errors such as missing processed data', async () => {
    respond({ detail: 'telemetry.parquet not found' }, 503)
    renderPanel()
    expect(await screen.findByRole('alert')).toHaveTextContent('telemetry.parquet not found')
  })

  it('shows stopped windows as explained and novel evidence without an absurd z-score', async () => {
    const windows = [
      { ...replay.windows[0], status: 'stopped' as const, score: null, exceedance: null },
      {
        ...replay.windows[2],
        evidence: [
          {
            group: 'protection_switches',
            group_label: 'Protection switches',
            feature: 'Oil_level_mean',
            value: 0,
            training_median: 1,
            robust_z: null,
            novel: true,
            meaning: 'Oil-level switch, active when the oil is below the expected level.',
          },
        ],
      },
      replay.windows[3],
    ]
    respond({
      ...replay,
      windows,
      episodes: [{ ...replay.episodes[0], ongoing: true, truncated: true }],
    })
    renderPanel()
    expect(await screen.findByText(/1 machine stopped \(explained\)/)).toBeInTheDocument()
    expect(screen.getByText('ongoing')).toBeInTheDocument()
    expect(screen.getByText(/started earlier/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Show evidence' }))
    expect(screen.getByText('never varied in training')).toBeInTheDocument()
    expect(screen.getByText('Protection switches')).toBeInTheDocument()
  })
})
