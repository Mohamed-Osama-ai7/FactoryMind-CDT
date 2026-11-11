import { createFileRoute, Link } from '@tanstack/react-router'
import { describeApiError } from '../api/errors'
import { useListIncidents } from '../api/generated/incidents/incidents'
import { formatNaive } from '../lib/naiveTime'

export const Route = createFileRoute('/incidents/')({
  component: IncidentsPage,
})

function IncidentsPage() {
  const incidents = useListIncidents()
  // non-2xx responses throw, so a resolved response is the 200 shape
  const list = incidents.data?.status === 200 ? incidents.data.data : undefined

  return (
    <section aria-labelledby="incidents-title" className="max-w-5xl">
      <h1 id="incidents-title" className="text-2xl font-semibold">
        Incidents
      </h1>
      <p className="mt-1 text-sm text-muted">
        Alarm episodes recorded for investigation and human review. Open an incident to inspect its
        multi-phase evidence package, quality checks, and chronological timeline. FactoryMind never
        acts on equipment.
      </p>
      <div className="mt-4">
        {incidents.isPending && <p role="status">Loading incidents…</p>}
        {incidents.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-sm">
            Could not load incidents: {describeApiError(incidents.error)}
          </p>
        )}
        {list && list.items.length === 0 && (
          <p role="status" className="text-sm text-muted">
            No incidents recorded yet. Create one from an alarm episode in the anomaly monitor.
          </p>
        )}
        {list && list.items.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="mb-1 text-left text-muted">
                {list.total} incident{list.total === 1 ? '' : 's'}
              </caption>
              <thead>
                <tr className="text-left text-muted">
                  {[
                    'Raised at',
                    'Asset',
                    'Detector',
                    'Status',
                    'Version',
                    'Created by',
                    'Action',
                  ].map((h) => (
                    <th key={h} scope="col" className="pr-4 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {list.items.map((i) => (
                  <tr key={i.id} className="border-t border-border">
                    <td className="py-1.5 pr-4">{formatNaive(i.raised_at)}</td>
                    <td className="py-1.5 pr-4">{i.asset_id}</td>
                    <td className="py-1.5 pr-4">{i.detector}</td>
                    <td className="py-1.5 pr-4">{i.status}</td>
                    <td className="py-1.5 pr-4">{i.version}</td>
                    <td className="py-1.5 pr-4">{i.created_by}</td>
                    <td className="py-1.5">
                      <Link
                        to="/incidents/$incidentId"
                        params={{ incidentId: i.id }}
                        className="text-accent underline"
                      >
                        Investigate
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  )
}
