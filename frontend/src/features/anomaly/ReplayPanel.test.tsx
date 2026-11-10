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
      evidence: [],
    },
    {
      window_start: '2020-06-05T09:10:00',
      window_end: '2020-06-05T09:20:00',
      n_samples: 12,
      status: 'insufficient_data',
      score: null,
      evidence: [],
    },
    {
      window_start: '2020-06-05T09:20:00',
      window_end: '2020-06-05T09:30:00',
      n_samples: 60,
      status: 'anomalous',
      score: 0.8,
      evidence: [{ feature: 'load_duty', value: 0.9, training_median: 0.1, robust_z: 12.5 }],
    },
    {
      window_start: '2020-06-05T09:30:00',
      window_end: '2020-06-05T09:40:00',
      n_samples: 60,
      status: 'anomalous',
      score: 0.75,
      evidence: [{ feature: 'TP2_mean', value: 9.1, training_median: 0.0, robust_z: 4.0 }],
    },
  ],
  episodes: [
    {
      start: '2020-06-05T09:20:00',
      end: '2020-06-05T09:40:00',
      raised_at: '2020-06-05T09:40:00',
      peak_score: 0.8,
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

function respond(body: unknown, status = 200) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(body), {
          status,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    ),
  )
}

describe('ReplayPanel', () => {
  it('requests the computed range and summarises windows, episodes and reports', async () => {
    respond(replay)
    renderPanel()
    expect(await screen.findByTestId('score-chart')).toHaveTextContent('4 windows')
    expect(
      screen.getByText(/1 normal, 2 above threshold, 1 with insufficient data/),
    ).toBeInTheDocument()
    expect(screen.getByText(/F3 Air leak/)).toBeInTheDocument()
    const url = String(vi.mocked(fetch).mock.calls[0]?.[0])
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
})
