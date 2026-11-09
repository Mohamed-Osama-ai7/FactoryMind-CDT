import { createFileRoute } from '@tanstack/react-router'
import { SystemStatus } from '../features/system/SystemStatus'

export const Route = createFileRoute('/')({
  component: OverviewPage,
})

function OverviewPage() {
  return (
    <section aria-labelledby="overview-title" className="max-w-3xl">
      <h1 id="overview-title" className="text-2xl font-semibold">
        Overview
      </h1>
      <p className="mt-1 text-sm text-muted">
        Research prototype on public and simulated data. Recommendations require human review;
        FactoryMind never controls equipment.
      </p>
      <div className="mt-6">
        <SystemStatus />
      </div>
    </section>
  )
}
