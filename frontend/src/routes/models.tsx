import { createFileRoute } from '@tanstack/react-router'
import { ModelsWorkspace } from '../features/ml/ModelsWorkspace'

export const Route = createFileRoute('/models')({
  component: ModelsPage,
})

function ModelsPage() {
  return <ModelsWorkspace />
}
