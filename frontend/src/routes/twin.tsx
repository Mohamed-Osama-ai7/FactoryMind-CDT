import { createFileRoute } from '@tanstack/react-router'
import { DigitalTwinLab } from '../features/twin/DigitalTwinLab'

export const Route = createFileRoute('/twin')({
  component: TwinPage,
})

function TwinPage() {
  return <DigitalTwinLab />
}
