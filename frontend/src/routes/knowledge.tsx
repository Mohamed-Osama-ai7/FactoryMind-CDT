import { createFileRoute } from '@tanstack/react-router'
import { KnowledgeExplorer } from '../features/knowledge/KnowledgeExplorer'

export const Route = createFileRoute('/knowledge')({
  component: KnowledgePage,
})

function KnowledgePage() {
  return <KnowledgeExplorer />
}
