import { useQueryClient } from '@tanstack/react-query'
import { describeApiError } from '../../api/errors'
import {
  getListIncidentsQueryKey,
  useCreateIncident,
  useMe,
} from '../../api/generated/incidents/incidents'

/** Asset whose telemetry the anomaly monitor replays (MetroPT-3 APU). */
export const ANOMALY_ASSET_ID = 'metropt3-apu-1'

export interface CreateIncidentButtonProps {
  rangeStart: string
  rangeEnd: string
  episodeStart: string
}

/** Records an alarm episode as an incident (engineers and admins). The server re-detects it. */
export function CreateIncidentButton({
  rangeStart,
  rangeEnd,
  episodeStart,
}: CreateIncidentButtonProps) {
  const me = useMe()
  const queryClient = useQueryClient()
  const create = useCreateIncident({
    mutation: {
      onSuccess: () => queryClient.invalidateQueries({ queryKey: getListIncidentsQueryKey() }),
    },
  })

  const profile = me.data?.data
  // The replayed telemetry belongs to one asset (the backend's data source); the caller's first
  // granted asset is not necessarily that one when several assets are granted.
  const assetId = profile?.asset_ids.includes(ANOMALY_ASSET_ID) ? ANOMALY_ASSET_ID : undefined
  if (!profile?.permissions.includes('incident:create') || !assetId) return null

  const result = create.data
  // the generated type only declares 201/422; the API answers 200 for an already-recorded incident
  if (create.isSuccess && result && Number(result.status) < 300) {
    const status = 'status' in result.data ? result.data.status : ''
    return (
      <span role="status" className="text-xs text-muted">
        {Number(result.status) === 201 ? 'Incident created' : 'Incident already recorded'} ({status}
        )
      </span>
    )
  }
  return (
    <span className="flex flex-col items-start">
      <button
        type="button"
        className="text-accent underline disabled:opacity-50"
        disabled={create.isPending}
        onClick={() =>
          create.mutate({
            data: {
              asset_id: assetId,
              range_start: rangeStart,
              range_end: rangeEnd,
              episode_start: episodeStart,
            },
          })
        }
      >
        Create incident
      </button>
      {create.isError && (
        <span role="alert" className="text-xs text-critical">
          {describeApiError(create.error)}
        </span>
      )}
    </span>
  )
}
