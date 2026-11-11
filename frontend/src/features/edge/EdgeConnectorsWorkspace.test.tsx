import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { EdgeConnectorsWorkspace } from './EdgeConnectorsWorkspace'

function mockMeOk() {
  return new Response(
    JSON.stringify({
      actor_id: 'eng-1',
      tenant_id: 'acme',
      roles: ['engineer'],
      permissions: ['asset:read', 'incident:read'],
      asset_ids: ['metropt3-apu-1'],
    }),
    { status: 200, headers: { 'Content-Type': 'application/json' } },
  )
}

describe('EdgeConnectorsWorkspace', () => {
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
        <EdgeConnectorsWorkspace />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/do not hold the/)).toBeInTheDocument()
  })

  it('renders error alerts when permission, sources, or hardening requests fail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Auth error' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { unmount } = render(
      <QueryClientProvider client={client}>
        <EdgeConnectorsWorkspace />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Could not verify OT\/edge permissions/,
    )
    unmount()

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Backend offline' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }),
    )

    const client2 = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client2}>
        <EdgeConnectorsWorkspace />
      </QueryClientProvider>,
    )
    expect(await screen.findByText(/Could not load OT sources/)).toBeInTheDocument()
    expect(await screen.findByText(/Could not load supply-chain report/)).toBeInTheDocument()
  })

  it('renders sources, supply-chain hardening, validates inputs, and runs batch check', async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
      if (url.endsWith('/api/v1/edge/sources')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              advisory_only: true,
              write_capabilities: 'none_disabled_by_architecture',
              sources: [
                {
                  source_id: 'metropt3-opcua-ro',
                  asset_id: 'metropt3-apu-1',
                  protocol: 'opcua_readonly',
                  endpoint_url: 'opc.tcp://opc-ro.factorymind.internal:4840',
                  read_only: true,
                  connection_state: 'standby_offline',
                  nominal_cadence_s: 10,
                  max_buffer_records: 512,
                  node_allowlist_count: 7,
                  ssrf_policy_status: 'allowlisted_host_verified',
                  buffer_utilization: 0,
                  reconnect_attempts: 0,
                },
              ],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }
      if (url.endsWith('/api/v1/edge/hardening')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              advisory_only: true,
              license_present: true,
              ci_actions_sha_pinned: true,
              unpinned_workflow_uses: [],
              non_pickle_attestation_ready: true,
              attestations: [
                {
                  name: 'GitHub Actions CI Workflow',
                  relative_path: '.github/workflows/ci.yml',
                  format: 'yaml_workflow',
                  status: 'sha_pinned',
                  sha256: 'abc123',
                  detail: 'All 4 CI workflow action steps pinned to 40-hex commit SHAs',
                },
              ],
              cross_system_benchmarks: {
                'M2-WP0/WP2 MetroPT-3 LOEO Detection': 'duty_rule 4/4 leaks',
              },
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({
            source_id: 'metropt3-opcua-ro',
            asset_id: 'metropt3-apu-1',
            protocol: 'opcua_readonly',
            read_only: true,
            endpoint_allowlist_ok: true,
            cadence: {
              sample_count: 6,
              expected_cadence_s: 10,
              observed_median_interval_s: 10,
              min_interval_s: 10,
              max_interval_s: 40,
              out_of_order_count: 0,
              duplicate_timestamp_count: 0,
              gap_count: 1,
              cadence_status: 'gapped_or_jittered',
              unallowlisted_sensors_ignored: ['UNALLOWLISTED_SETPOINT_WRITE'],
            },
            buffer: {
              capacity: 4,
              retained_count: 4,
              evicted_count: 2,
              reconnect_attempts: 0,
              backoff_schedule_s: [],
              state: 'degraded_buffer_evicting',
            },
            warnings: ['Detected 1 sampling gap(s) exceeding 20.0s; preserved uninterpolated.'],
            safety_note: 'Read-only advisory edge validation only.',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      )
    })
    vi.stubGlobal('fetch', fetchMock)

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const user = userEvent.setup()
    render(
      <QueryClientProvider client={client}>
        <EdgeConnectorsWorkspace />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/allowlisted_host_verified/)).toBeInTheDocument()
    expect(screen.getByText(/CI actions SHA-pinned: yes/)).toBeInTheDocument()

    // Invalid buffer capacity must show scoped aria-invalid, aria-describedby, and block POST
    const callsBeforeValidation = fetchMock.mock.calls.length
    const capInput = screen.getByLabelText(/Buffer capacity/)
    await user.clear(capInput)
    await user.type(capInput, '2')
    await user.click(screen.getByRole('button', { name: /Run Read-Only Batch Check/ }))
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/Buffer capacity must be an integer between 4 and 10000/)
    expect(capInput).toHaveAttribute('aria-invalid', 'true')
    expect(capInput).toHaveAttribute('aria-describedby', alert.id)
    expect(screen.getByLabelText(/Nominal step/)).toHaveAttribute('aria-invalid', 'false')
    expect(fetchMock).toHaveBeenCalledTimes(callsBeforeValidation)

    // Typing clears prior validation error cleanly
    await user.clear(capInput)
    await user.type(capInput, '4')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Run Read-Only Batch Check/ }))

    expect(await screen.findByText(/cadence: gapped_or_jittered/)).toBeInTheDocument()
    expect(screen.getByText(/retained 4\/4 · evicted 2/)).toBeInTheDocument()

    // Failed validation POST renders mutation error alert
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'Simulated server rejection' }), {
        status: 422,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    await user.click(screen.getByRole('button', { name: /Run Read-Only Batch Check/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Simulated server rejection/)
  })
})
