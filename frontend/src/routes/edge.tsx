import { createFileRoute } from '@tanstack/react-router'
import { EdgeConnectorsWorkspace } from '../features/edge/EdgeConnectorsWorkspace'

export const Route = createFileRoute('/edge')({
  component: EdgePage,
})

function EdgePage() {
  return <EdgeConnectorsWorkspace />
}
