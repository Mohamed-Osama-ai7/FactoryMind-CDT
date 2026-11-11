import type {
  ExceedanceCheck,
  FaultSummary,
  LoeoSummary,
  ModelInfo,
  ReferenceHoldoutCheck,
  Wp2EvaluationSummary,
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

function Wp2RegimesAndWarnings({ wp2 }: { wp2: Wp2EvaluationSummary }) {
  const sep = wp2.event_separation
  const regimeKeys = ['loaded', 'unloaded', 'starting', 'stopped', 'insufficient_evidence']
  const ablations = [
    ['wp0_stopped_only', 'WP0 baseline (stopped suppressed)'],
    ['wp2_integrity_gated', 'WP2 integrity-gated (stopped + stale + inconsistent)'],
    ['starting_suppressed_ablation', 'Ablation: also suppress starting'],
  ] as const

  return (
    <div className="mt-4 flex flex-col gap-4 rounded border border-border p-3 text-sm">
      <div>
        <h3 className="font-medium">Operating-regime &amp; warning separation (WP2)</h3>
        <p className="mt-1 text-muted">
          Primary leak alarms ({wp2.primary_detector}) abstain on frozen or contradictory telemetry:{' '}
          <span className="font-medium text-ink">
            {sep.integrity_suppressed_leak_episodes} false leak episode
            {sep.integrity_suppressed_leak_episodes === 1 ? '' : 's'} prevented
          </span>{' '}
          ({sep.gated_false_alarm_episodes_wp2} gated vs {sep.baseline_false_alarm_episodes_wp0} WP0
          false alarms). Consistency warnings ({wp2.warning_detector ?? 'signal integrity'}):{' '}
          <span className="font-medium text-ink">
            {sep.sensor_consistency_warning_episodes} episode
            {sep.sensor_consistency_warning_episodes === 1 ? '' : 's'}
          </span>{' '}
          ({sep.warnings_per_day.toFixed(3)} / valid day, {sep.warning_only_episodes} standalone,{' '}
          {sep.warning_overlapping_leak_episodes} overlapping leak episodes).
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="mb-1 text-left text-xs text-muted">
            Per-regime evaluation across valid windows (independent primary air-leak vs consistency
            warning rates)
          </caption>
          <thead>
            <tr className="text-left text-muted">
              <th scope="col" className="pr-3 font-normal">
                Regime
              </th>
              <th scope="col" className="pr-3 font-normal">
                Windows
              </th>
              <th scope="col" className="pr-3 font-normal">
                Primary exceedances
              </th>
              <th scope="col" className="pr-3 font-normal">
                Leak FA episodes
              </th>
              <th scope="col" className="font-normal">
                Warning exceedances / episodes
              </th>
            </tr>
          </thead>
          <tbody>
            {regimeKeys.map((key) => {
              const row = wp2.per_regime_metrics[key]
              if (!row) return null
              return (
                <tr key={key} className="border-t border-border">
                  <th scope="row" className="py-1 pr-3 text-left font-normal">
                    {key}
                  </th>
                  <td className="py-1 pr-3">
                    {row.windows} ({pct(row.share)})
                  </td>
                  <td className="py-1 pr-3">
                    {row.primary.scored_windows > 0
                      ? `${row.primary.exceedance_windows}/${row.primary.scored_windows} (${pct(row.primary.exceedance_rate)})`
                      : 'abstained'}
                  </td>
                  <td className="py-1 pr-3">
                    {row.primary.false_alarm_episodes} (
                    {row.primary.overlapping_false_alarm_episodes} overlapping)
                  </td>
                  <td className="py-1">
                    {row.warning
                      ? `${row.warning.exceedance_windows}/${row.warning.scored_windows} (${pct(row.warning.exceedance_rate)}) · ${row.warning.warning_episodes} ep`
                      : 'n/a'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="mb-1 text-left text-xs text-muted">
            LOEO regime-gating ablations on {wp2.primary_detector}
          </caption>
          <thead>
            <tr className="text-left text-muted">
              <th scope="col" className="pr-3 font-normal">
                Policy
              </th>
              <th scope="col" className="pr-3 font-normal">
                LOEO leaks
              </th>
              <th scope="col" className="pr-3 font-normal">
                FA / valid day
              </th>
              <th scope="col" className="pr-3 font-normal">
                FA / op day
              </th>
              <th scope="col" className="font-normal">
                Note
              </th>
            </tr>
          </thead>
          <tbody>
            {ablations.map(([key, label]) => {
              const abl = wp2.regime_gating_ablations[key]
              if (!abl) return null
              return (
                <tr key={key} className="border-t border-border align-top">
                  <th scope="row" className="py-1 pr-3 text-left font-normal">
                    {label}
                  </th>
                  <td className="py-1 pr-3">
                    {abl.events_detected}/{abl.events_total}
                  </td>
                  <td className="py-1 pr-3">{abl.mean_false_alarms_per_day.toFixed(3)}</td>
                  <td className="py-1 pr-3">
                    {abl.mean_operating_false_alarms_per_day.toFixed(3)}
                  </td>
                  <td className="py-1 text-xs text-muted">{abl.note}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-muted">
        {wp2.simulated_fault_separation.note}{' '}
        {Object.entries(wp2.simulated_fault_separation.detectors)
          .map(
            ([det, d]) =>
              `${det}: ${d.detected}/${d.cases} gross, ctrl ${d.control_detected}/${d.cases}, clean sens ${pct(d.clean_control_sensitivity)}, lift ${d.incremental_lift === null || d.incremental_lift === undefined ? 'n/a' : `${d.incremental_lift >= 0 ? '+' : ''}${(d.incremental_lift * 100).toFixed(1)} pp`}`,
          )
          .join(' · ')}
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
      {model.wp2 && <Wp2RegimesAndWarnings wp2={model.wp2} />}
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
