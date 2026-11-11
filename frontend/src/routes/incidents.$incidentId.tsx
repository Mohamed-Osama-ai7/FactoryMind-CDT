import { createFileRoute, Link } from '@tanstack/react-router'
import { describeApiError } from '../api/errors'
import {
  useGetIncident,
  useGetIncidentEvidencePackage,
  useGetIncidentRca,
} from '../api/generated/incidents/incidents'
import { DecisionComparisonPanel } from '../features/incidents/DecisionComparisonPanel'
import { DecisionPanel } from '../features/incidents/DecisionPanel'
import { EventEvidencePackageView } from '../features/incidents/EventEvidencePackageView'
import { ObservationForm } from '../features/incidents/ObservationForm'
import { OutcomeVerificationPanel } from '../features/incidents/OutcomeVerificationPanel'
import { RcaPanel } from '../features/incidents/RcaPanel'
import { formatNaive } from '../lib/naiveTime'

export const Route = createFileRoute('/incidents/$incidentId')({
  component: IncidentInvestigationPage,
})

function IncidentInvestigationPage() {
  const { incidentId } = Route.useParams()
  const incidentQuery = useGetIncident(incidentId)
  const evidenceQuery = useGetIncidentEvidencePackage(incidentId)
  const rcaQuery = useGetIncidentRca(incidentId)

  const incident = incidentQuery.data?.status === 200 ? incidentQuery.data.data : undefined
  const pkg = evidenceQuery.data?.status === 200 ? evidenceQuery.data.data : undefined
  const rca = rcaQuery.data?.status === 200 ? rcaQuery.data.data : undefined

  return (
    <section aria-labelledby="investigation-title" className="flex max-w-5xl flex-col gap-6">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p className="text-xs text-muted">
            <Link to="/incidents" className="text-accent underline">
              ← Back to incidents
            </Link>
          </p>
          <h1 id="investigation-title" className="mt-1 text-2xl font-semibold">
            Incident investigation
          </h1>
        </div>
        {incident && (
          <span className="rounded border border-border px-2.5 py-1 text-xs">
            Status: <strong>{incident.status}</strong> (v{incident.version})
          </span>
        )}
      </header>

      {incidentQuery.isPending && <p role="status">Loading incident…</p>}
      {incidentQuery.isError && (
        <p role="alert" className="rounded border border-critical p-3 text-sm">
          Could not load incident: {describeApiError(incidentQuery.error)}
        </p>
      )}

      {incident && (
        <section
          aria-labelledby="incident-metadata"
          className="rounded-lg border border-border bg-panel p-4 text-sm"
        >
          <h2 id="incident-metadata" className="font-medium">
            Incident record
          </h2>
          <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <dt className="text-muted">Asset</dt>
              <dd className="font-mono text-ink">{incident.asset_id}</dd>
            </div>
            <div>
              <dt className="text-muted">Event interval</dt>
              <dd>
                {formatNaive(incident.episode_start)} to {formatNaive(incident.episode_end)}
              </dd>
            </div>
            <div>
              <dt className="text-muted">Model &amp; detector</dt>
              <dd className="font-mono">
                {incident.detector} ({incident.model_version})
              </dd>
            </div>
            <div>
              <dt className="text-muted">Peak score &amp; creator</dt>
              <dd>
                {incident.peak_score.toFixed(3)} · {incident.created_by}
              </dd>
            </div>
          </dl>

          <div className="mt-4 border-t border-border pt-3">
            <h3 className="text-sm font-medium">
              Human field observations ({incident.observations?.length ?? 0})
            </h3>
            {(incident.observations?.length ?? 0) === 0 ? (
              <p className="mt-1 text-xs text-muted">
                No field observations recorded for this incident yet.
              </p>
            ) : (
              <ul className="mt-2 space-y-1.5 text-xs">
                {incident.observations?.map((obs) => (
                  <li key={obs.id} className="rounded border border-border p-2">
                    <span className="font-medium text-ink">
                      {obs.category} · {obs.actor_id}
                    </span>{' '}
                    <span className="text-muted">
                      ({formatNaive(obs.observed_at ?? obs.created_at)})
                    </span>
                    <p className="mt-0.5 text-ink">{obs.note}</p>
                  </li>
                ))}
              </ul>
            )}
            <ObservationForm
              key={incident.id}
              incidentId={incident.id}
              defaultTimestamp={incident.episode_start}
            />
          </div>
        </section>
      )}

      {incident && (
        <>
          <DecisionComparisonPanel
            key={`${incident.id}-comparison`}
            incidentId={incident.id}
            dominantHypothesis={rca?.primary_category}
          />
          <DecisionPanel key={`${incident.id}-decisions`} incident={incident} />
          <OutcomeVerificationPanel key={`${incident.id}-outcomes`} incident={incident} />
        </>
      )}

      {evidenceQuery.isPending && <p role="status">Building event evidence package…</p>}
      {evidenceQuery.isError && (
        <p role="alert" className="rounded border border-critical p-3 text-sm">
          Evidence package unavailable: {describeApiError(evidenceQuery.error)}
        </p>
      )}
      {pkg && <EventEvidencePackageView key={`${incidentId}-${pkg.package_id}`} pkg={pkg} />}
      {rcaQuery.isPending && <p role="status">Computing explainable RCA hypotheses…</p>}
      {rcaQuery.isError && (
        <p role="alert" className="rounded border border-critical p-3 text-sm">
          RCA unavailable: {describeApiError(rcaQuery.error)}
        </p>
      )}
      {rca && <RcaPanel key={`${incidentId}-${rca.rca_id}`} rca={rca} />}
    </section>
  )
}
