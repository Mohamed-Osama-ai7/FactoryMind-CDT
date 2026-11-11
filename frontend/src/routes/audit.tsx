import { createFileRoute } from '@tanstack/react-router'
import { AuditViewer } from '../features/audit/AuditViewer'

export const Route = createFileRoute('/audit')({
  component: AuditPage,
})

function AuditPage() {
  return <AuditViewer />
}
