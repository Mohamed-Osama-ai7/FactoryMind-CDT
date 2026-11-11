import { createFileRoute, Link } from '@tanstack/react-router'
import { AssetPassportDetail } from '../features/assets/AssetPassportDetail'

export const Route = createFileRoute('/assets/$assetId')({
  component: AssetPassportPage,
})

function AssetPassportPage() {
  const { assetId } = Route.useParams()
  return (
    <AssetPassportDetail
      assetId={assetId}
      backLink={
        <Link to="/assets" className="text-accent underline">
          ← Back to assets &amp; data foundation
        </Link>
      }
    />
  )
}
