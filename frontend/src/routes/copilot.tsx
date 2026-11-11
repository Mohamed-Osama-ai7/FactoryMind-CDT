import { createFileRoute } from '@tanstack/react-router'
import { CopilotWorkspace } from '../features/copilot/CopilotWorkspace'

export const Route = createFileRoute('/copilot')({
  component: CopilotPage,
})

function CopilotPage() {
  return <CopilotWorkspace />
}
