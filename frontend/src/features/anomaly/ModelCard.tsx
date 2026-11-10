import type {
  ExceedanceCheck,
  FaultSummary,
  LoeoSummary,
  ModelInfo,
  ReferenceHoldoutCheck,
} from '../../api/generated/model'

function minutes(value: number | null | undefined): string {
  return value === null || value === undefined ? 'n/a' : `${Math.round(value)} min`
}

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? 'n/a' : `${(value * 100).toFixed(1)} %`
}

function Loeo({ loeo }: { loeo: LoeoSummary }) {
  return (
    <table className="mt-4 w-full text-sm">
      <caption className="mb-1 text-left text-muted">
        Leave-one-event-out evaluation against operator failure reports (policy chosen
        non-chronologically on the other reports)
      </caption>
      <thead>
        <tr className="text-left text-muted">
          <th scope="col" className="pr-4 font-normal">
            Held-out report
          </th>
          <th scope="col" className="pr-4 font-normal">
            Detected
          </th>
          <th scope="col" className="font-normal">
            Delay after report start
          </th>
        </tr>
      </thead>
      <tbody>
        {loeo.folds.map((f) => (
          <tr key={f.held_out} className="border-t border-border">
            <th scope="row" className="py-1.5 pr-4 text-left font-normal">
              {f.held_out}
            </th>
            <td className="py-1.5 pr-4">{f.detected ? 'yes' : 'no'}</td>
            <td className="py-1.5">{f.detected ? minutes(f.delay_minutes) : 'n/a'}</td>
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr className="border-t border-border text-muted">
          <td className="py-1.5 pr-4">Total</td>
          <td className="py-1.5 pr-4">
            {loeo.events_detected}/{loeo.events_total}
          </td>
          <td className="py-1.5">
            {loeo.mean_false_alarms_per_day.toFixed(3)} false alarms / valid day (
            {loeo.mean_operating_false_alarms_per_day.toFixed(3)} false alarms / operating day) ·
            median |delay| {minutes(loeo.median_abs_delay_minutes)}
          </td>
        </tr>
      </tfoot>
    </table>
  )
}

function Faults({ faults }: { faults: FaultSummary }) {
  const lift =
    faults.incremental_lift === null || faults.incremental_lift === undefined
      ? 'n/a'
      : `${faults.incremental_lift >= 0 ? '+' : ''}${(faults.incremental_lift * 100).toFixed(1)} pp`
  return (
    <p className="mt-3 text-sm text-muted">
      Simulated sensor faults injected into real normal telemetry: gross detection {faults.detected}
      /{faults.cases} ({pct(faults.gross_rate)}), unmodified control alarms {faults.control_alarms}{' '}
      ({pct(faults.control_alarm_rate)}), clean-control sensitivity {faults.clean_control_detected}/
      {faults.clean_control_cases} ({pct(faults.clean_control_sensitivity)}), incremental lift{' '}
      {lift}.
    </p>
  )
}

function CalibrationCheck({
  holdout,
  exceedanceEval,
}: {
  holdout: ReferenceHoldoutCheck | null | undefined
  exceedanceEval: ExceedanceCheck | null | undefined
}) {
  if (!holdout && !exceedanceEval) return null
  return (
    <div className="mt-3 rounded border border-border p-3 text-sm">
      <p className="font-medium">Calibration reference check</p>
      {holdout && (
        <p className="mt-1 text-muted">
          Split-half holdout at nominal {pct(holdout.alpha)}:{' '}
          <span className="font-medium text-ink">{pct(holdout.share_at_or_below_alpha)}</span> of{' '}
          {holdout.windows} held-out calibration windows fell at or below nominal α.
        </p>
      )}
      {exceedanceEval && (
        <p className="mt-1 text-muted">
          Far-from-report operating windows at nominal {pct(exceedanceEval.alpha)}:{' '}
          {pct(exceedanceEval.share_at_or_below_alpha)} across {exceedanceEval.windows} windows.
        </p>
      )}
      <p className="mt-1 text-xs text-muted">
        Calibration limitation: normal operation drifts over time; exceedance is the share of
        calibration-reference windows scoring at least as high, never a failure probability.
      </p>
    </div>
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
        <dd>
          {model.detector}
          {model.selected_as_default ? ' (default)' : ''}
        </dd>
        <dt className="text-muted">Why this detector</dt>
        <dd>{model.selection_rationale}</dd>
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
          {model.window_length} windows. A reference-normal window reaches the threshold about{' '}
          {(model.exceedance_at_threshold * 100).toFixed(1)} % of the time.
        </dd>
      </dl>
      <Loeo loeo={model.loeo} />
      {model.fault_injection && <Faults faults={model.fault_injection} />}
      <CalibrationCheck
        holdout={model.reference_holdout_check}
        exceedanceEval={model.exceedance_check}
      />
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
