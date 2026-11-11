import { createFileRoute, Link } from '@tanstack/react-router'
import { AssetsOverview } from '../features/assets/AssetsOverview'

export const Route = createFileRoute('/assets/')({
  component: AssetsPage,
})

function AssetsPage() {
  return (
    <AssetsOverview
      renderOpenLink={(assetId) => (
        <Link to="/assets/$assetId" params={{ assetId }} className="text-xs text-accent underline">
          Open digital passport →
        </Link>
      )}
    />
  )
}
