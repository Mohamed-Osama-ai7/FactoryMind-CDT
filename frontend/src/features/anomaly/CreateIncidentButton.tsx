import { useQueryClient } from '@tanstack/react-query'
import { describeApiError } from '../../api/errors'
import {
  getListIncidentsQueryKey,
  useCreateIncident,
  useMe,
} from '../../api/generated/incidents/incidents'

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
  const assetId = profile?.asset_ids[0]
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
