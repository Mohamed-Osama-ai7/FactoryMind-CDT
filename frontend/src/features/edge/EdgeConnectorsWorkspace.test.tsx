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

  it('renders loading, permission errors, and source/hardening failure states accurately', async () => {
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(new Promise<Response>(() => {})))
    const pendingClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { unmount: unmountPending } = render(
      <QueryClientProvider client={pendingClient}>
        <EdgeConnectorsWorkspace />
      </QueryClientProvider>,
    )
    expect(screen.getByRole('status')).toHaveTextContent(/Checking edge preflight permissions/)
    unmountPending()

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
      /Could not verify edge preflight permissions/,
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
    expect(await screen.findByText(/Could not load edge source descriptors/)).toBeInTheDocument()
    expect(await screen.findByText(/Could not load supply-chain check report/)).toBeInTheDocument()
  })

  it('renders unavailable/empty descriptors and failed attestation states honestly', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
        if (url.endsWith('/api/v1/edge/sources')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                advisory_only: true,
                connector_io_status: 'not_implemented',
                write_capabilities: 'none_no_connector_io_exists',
                capability_note:
                  'These are registered source descriptors. No live OPC-UA, MQTT or historian I/O exists.',
                sources: [],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              advisory_only: true,
              license_present: false,
              ci_actions_sha_pinned: false,
              unpinned_workflow_uses: ['actions/checkout@v4'],
              runtime_model_attestation: 'integrity_failed',
              runtime_model_format: 'joblib_pickle',
              runtime_loader_consults_manifest: false,
              manifest_signature_key_configured: false,
              attestations: [
                {
                  name: 'Root Repository License',
                  relative_path: 'LICENSE',
                  format: 'text_license',
                  status: 'missing',
                  sha256: null,
                  detail: 'Root LICENSE file missing',
                },
                {
                  name: 'Model v1 joblib bundle',
                  relative_path: 'artifacts/metropt3-anomaly/v1/model.joblib',
                  format: 'joblib_legacy',
                  status: 'tamper_or_hash_mismatch',
                  sha256: 'deadbeef',
                  detail: 'SHA-256 does not match model_card.json',
                },
              ],
              limitations: ['No manifest signature is verified because no HMAC key is configured.'],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <EdgeConnectorsWorkspace />
      </QueryClientProvider>,
    )

    expect(
      await screen.findByText(
        /No registered source descriptors available for your asset permissions/,
      ),
    ).toBeInTheDocument()
    expect(screen.getByText(/LICENSE: missing\/invalid/)).toBeInTheDocument()
    expect(screen.getByText(/CI uses syntax: unpinned/)).toBeInTheDocument()
    expect(screen.getByText(/runtime model: integrity_failed/)).toBeInTheDocument()
    expect(screen.getByText(/Unpinned CI actions: actions\/checkout@v4/)).toBeInTheDocument()
    expect(screen.getByText(/loader consults manifest: no/)).toBeInTheDocument()
    expect(screen.getByText(/HMAC key configured: no/)).toBeInTheDocument()
    expect(
      screen.getByText(/No manifest signature is verified because no HMAC key is configured/),
    ).toBeInTheDocument()
  })

  it('renders descriptor-only status, unverified checks, validates inputs, and runs preflight batch check', async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
      if (url.endsWith('/api/v1/edge/sources')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              advisory_only: true,
              connector_io_status: 'not_implemented',
              write_capabilities: 'none_no_connector_io_exists',
              capability_note:
                'These are registered source descriptors. FactoryMind does not currently connect to any OPC-UA server, MQTT broker or historian; endpoint checks inspect the URL text only.',
              sources: [
                {
                  source_id: 'metropt3-opcua-ro',
                  asset_id: 'metropt3-apu-1',
                  protocol: 'opcua_readonly',
                  endpoint_url: 'opc.tcp://opc-ro.factorymind.internal:4840',
                  read_only: true,
                  connection_state: 'descriptor_only_not_connected',
                  connector_io: 'not_implemented',
                  nominal_cadence_s: 10,
                  max_buffer_records: 512,
                  node_allowlist_count: 7,
                  endpoint_url_policy: 'passed',
                  endpoint_check_scope: 'url_string_only_no_dns_resolution',
                },
                {
                  source_id: 'rejected-test-source',
                  asset_id: 'metropt3-apu-1',
                  protocol: 'mqtt_readonly',
                  endpoint_url: 'mqtt://plain.invalid:1883',
                  read_only: true,
                  connection_state: 'descriptor_only_not_connected',
                  connector_io: 'not_implemented',
                  nominal_cadence_s: 10,
                  max_buffer_records: 512,
                  node_allowlist_count: 1,
                  endpoint_url_policy: 'rejected',
                  endpoint_check_scope: 'url_string_only_no_dns_resolution',
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
              runtime_model_attestation: 'manifest_unsigned_self_consistent',
              runtime_model_format: 'joblib_pickle',
              runtime_loader_consults_manifest: false,
              manifest_signature_key_configured: false,
              attestations: [
                {
                  name: 'GitHub Actions CI Workflow',
                  relative_path: '.github/workflows/ci.yml',
                  format: 'yaml_workflow',
                  status: 'sha_pinned',
                  sha256: 'abc123',
                  detail: "All 4 'uses:' lines in ci.yml name a 40-hex commit",
                },
                {
                  name: 'Model v1 JSON manifest',
                  relative_path: 'artifacts/metropt3-anomaly/v1/model_manifest.json',
                  format: 'json_manifest',
                  status: 'self_consistent_unsigned',
                  sha256: 'def456',
                  detail: 'Parameters match own digest; no signature verified',
                },
              ],
              limitations: [
                'No runtime code path reads model_manifest.json.',
                'The runtime loads model.joblib with joblib (pickle).',
              ],
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
            mode: 'preflight_validation_only',
            real_connector_io: false,
            endpoint_url_policy_ok: true,
            endpoint_check_scope: 'url_string_only_no_dns_resolution',
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
              simulated_reconnect_attempts: 0,
              backoff_schedule_s: [],
              state: 'batch_exceeds_capacity',
            },
            warnings: ['Detected 1 sampling gap(s) exceeding 20.0s; preserved uninterpolated.'],
            safety_note: 'Preflight validation of the submitted batch only. No connector I/O.',
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

    const stateCells = await screen.findAllByText(
      /descriptor_only_not_connected \(not_implemented\)/,
    )
    expect(stateCells).toHaveLength(2)
    expect(screen.getByText('passed')).toBeInTheDocument()
    expect(screen.getByText('rejected')).toBeInTheDocument()
    expect(screen.getByText(/runtime model: manifest_unsigned_self_consistent/)).toBeInTheDocument()
    expect(screen.getByText('self_consistent_unsigned')).toBeInTheDocument()
    expect(screen.getByText(/No runtime code path reads model_manifest.json/)).toBeInTheDocument()

    // Invalid buffer capacity must show scoped aria-invalid, aria-describedby, and block POST
    const callsBeforeValidation = fetchMock.mock.calls.length
    const capInput = screen.getByLabelText(/Buffer capacity/)
    await user.clear(capInput)
    await user.type(capInput, '2')
    await user.click(screen.getByRole('button', { name: /Run Preflight Batch Check/ }))
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

    await user.click(screen.getByRole('button', { name: /Run Preflight Batch Check/ }))

    expect(await screen.findByText(/mode: preflight_validation_only/)).toBeInTheDocument()
    expect(screen.getByText(/buffer state: batch_exceeds_capacity/)).toBeInTheDocument()
    expect(screen.getByText(/retained 4\/4 · evicted 2 · sim reconnects 0/)).toBeInTheDocument()

    // Failed validation POST renders mutation error alert
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'Simulated server rejection' }), {
        status: 422,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    await user.click(screen.getByRole('button', { name: /Run Preflight Batch Check/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Simulated server rejection/)
  })
})
