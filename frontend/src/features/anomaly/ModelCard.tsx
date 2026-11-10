import type { ModelInfo, SplitMetrics } from '../../api/generated/model'

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? 'n/a' : value.toFixed(2)
}

function MetricsRow({ label, m }: { label: string; m: SplitMetrics }) {
  return (
    <tr className="border-t border-border">
      <th scope="row" className="py-1.5 pr-4 text-left font-normal text-muted">
        {label}
      </th>
      <td className="py-1.5 pr-4">
        {m.events_detected}/{m.events_total}
      </td>
      <td className="py-1.5 pr-4">{m.false_alarms_per_day.toFixed(3)}</td>
      <td className="py-1.5 pr-4">{pct(m.average_precision)}</td>
      <td className="py-1.5">{m.valid_days.toFixed(0)}</td>
    </tr>
  )
}

/** Summary of the deployed model: provenance, alarm policy and held-out evaluation. */
export function ModelCard({ model }: { model: ModelInfo }) {
  return (
    <section aria-labelledby="model-title" className="rounded-lg border border-border bg-panel p-4">
      <h2 id="model-title" className="font-medium">
        Model
      </h2>
      <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm">
        <dt className="text-muted">Version</dt>
        <dd className="font-mono">{model.model_version}</dd>
        <dt className="text-muted">Detector</dt>
        <dd>{model.detector}</dd>
        <dt className="text-muted">Dataset</dt>
        <dd>
          <a className="text-accent underline" href={model.dataset_source}>
            {model.dataset_id}
          </a>{' '}
          ({model.dataset_license}, {model.data_kind})
        </dd>
        <dt className="text-muted">Trained on</dt>
        <dd>
          {model.training_period[0]} to {model.training_period[1]}
        </dd>
        <dt className="text-muted">Alarm policy</dt>
        <dd>
          score &gt; {model.threshold.toFixed(3)} for {model.persistence} consecutive{' '}
          {model.window_length} windows
        </dd>
      </dl>
      <table className="mt-4 w-full text-sm">
        <caption className="mb-1 text-left text-muted">
          Held-out evaluation against operator failure reports
        </caption>
        <thead>
          <tr className="text-left text-muted">
            <th scope="col" className="pr-4 font-normal">
              Split
            </th>
            <th scope="col" className="pr-4 font-normal">
              Reports detected
            </th>
            <th scope="col" className="pr-4 font-normal">
              False alarms/day
            </th>
            <th scope="col" className="pr-4 font-normal">
              Window PR-AUC
            </th>
            <th scope="col" className="font-normal">
              Days
            </th>
          </tr>
        </thead>
        <tbody>
          <MetricsRow label="Validation" m={model.validation} />
          <MetricsRow label="Test" m={model.test} />
        </tbody>
      </table>
      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-muted">Intended use and limitations</summary>
        <p className="mt-2">{model.intended_use}</p>
        <ul className="mt-2 list-disc pl-5 text-muted">
          {model.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </details>
    </section>
  )
}
