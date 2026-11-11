import { LineChart, ScatterChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'
import type {
  AlarmEpisode,
  OperatorReport,
  WarningEpisode,
  WindowResult,
} from '../../api/generated/model'
import { formatNaive, toEpochMs } from '../../lib/naiveTime'

echarts.use([
  LineChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  DataZoomComponent,
  MarkLineComponent,
  MarkAreaComponent,
  CanvasRenderer,
])

export interface ScoreChartProps {
  windows: WindowResult[]
  threshold: number
  episodes: AlarmEpisode[]
  reports: OperatorReport[]
  warnings?: WarningEpisode[]
}

const COLORS = {
  line: '#3b82f6',
  threshold: '#f59e0b',
  alarm: 'rgba(239, 68, 68, 0.22)',
  warning: 'rgba(56, 189, 248, 0.16)',
  warningPoint: '#38bdf8',
  report: 'rgba(245, 158, 11, 0.14)',
  axis: '#8a96ad',
  grid: '#1f2a40',
}

const EVENT_CLASS_LABELS: Record<string, string> = {
  air_leak_alarm: 'Air-leak alarm',
  sensor_consistency_warning: 'Sensor-consistency warning',
  informational: 'Informational',
  insufficient_evidence: 'Insufficient evidence',
}

/** Anomaly score per window vs threshold, separating leak alarms from consistency warnings.
 * Missing or integrity-abstained windows leave gaps on the score line (never interpolated). */
export function ScoreChart({
  windows,
  threshold,
  episodes,
  reports,
  warnings = [],
}: ScoreChartProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const element = ref.current
    if (!element) return
    const chart = echarts.init(element, undefined, { renderer: 'canvas' })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element)
    return () => {
      observer.disconnect()
      chart.dispose()
    }
  }, [])

  useEffect(() => {
    const element = ref.current
    if (!element) return
    const chart = echarts.getInstanceByDom(element)
    if (!chart) return
    const windowByEpoch = new Map<number, WindowResult>(
      windows.map((w) => [toEpochMs(w.window_start), w]),
    )
    const warningPoints = windows
      .filter((w) => w.warning_active)
      .map((w) => [toEpochMs(w.window_start), w.score ?? threshold])

    chart.setOption(
      {
        animation: false,
        grid: { left: 56, right: 24, top: 24, bottom: 56 },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            label: {
              formatter: (p: { value: unknown }) =>
                formatNaive(new Date(Number(p.value)).toISOString()),
            },
          },
          formatter: (params: unknown) => {
            const first = Array.isArray(params) ? params[0] : params
            const datum =
              first && typeof first === 'object' && 'value' in first
                ? (first as { value?: unknown }).value
                : undefined
            const ts = Array.isArray(datum) ? Number(datum[0]) : Number.NaN
            const win = Number.isFinite(ts) ? windowByEpoch.get(ts) : undefined
            if (!win) return ''
            const scoreStr = win.score === null ? 'abstained / n/a' : win.score.toFixed(3)
            const eventLabel = EVENT_CLASS_LABELS[win.event_class] ?? win.event_class
            const regLine = `${win.regime.regime} (${win.regime.confidence})`
            const lines = [
              `<strong>${formatNaive(win.window_start)}</strong>`,
              `Duty score: ${scoreStr}`,
              `Event class: ${eventLabel}`,
              `Regime: ${regLine}`,
            ]
            if (win.warning_active) {
              const reasonText = win.warning_reasons?.join(', ') || win.regime.reason
              lines.push(`Warning: ${reasonText}`)
            }
            return lines.join('<br/>')
          },
        },
        xAxis: {
          type: 'time',
          axisLabel: {
            color: COLORS.axis,
            hideOverlap: true,
            // "MM-DD HH:mm" in dataset time (epoch values encode naive time as UTC)
            formatter: (ms: number) => formatNaive(new Date(ms).toISOString()).slice(5),
          },
          splitNumber: 6,
          axisLine: { lineStyle: { color: COLORS.grid } },
        },
        yAxis: {
          type: 'value',
          name: 'duty score',
          nameTextStyle: { color: COLORS.axis },
          axisLabel: { color: COLORS.axis },
          splitLine: { lineStyle: { color: COLORS.grid } },
          scale: true,
        },
        dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 12 }],
        series: [
          {
            name: 'score',
            type: 'line',
            showSymbol: false,
            connectNulls: false,
            lineStyle: { color: COLORS.line, width: 1.5 },
            itemStyle: { color: COLORS.line },
            data: windows.map((w) => [toEpochMs(w.window_start), w.score ?? null]),
            markLine: {
              symbol: 'none',
              silent: true,
              lineStyle: { color: COLORS.threshold, type: 'dashed' },
              label: { formatter: 'threshold', color: COLORS.threshold },
              data: [{ yAxis: threshold }],
            },
            markArea: {
              silent: true,
              data: [
                ...warnings.map((w) => [
                  {
                    xAxis: toEpochMs(w.start),
                    itemStyle: { color: COLORS.warning },
                  },
                  { xAxis: toEpochMs(w.end) },
                ]),
                ...episodes.map((e) => [
                  { xAxis: toEpochMs(e.start), itemStyle: { color: COLORS.alarm } },
                  { xAxis: toEpochMs(e.end) },
                ]),
                ...reports.map((r) => [
                  {
                    xAxis: toEpochMs(r.start),
                    itemStyle: { color: COLORS.report },
                    label: { show: true, formatter: r.report_id, color: COLORS.threshold },
                  },
                  { xAxis: toEpochMs(r.end) },
                ]),
              ],
            },
          },
          {
            name: 'consistency warning',
            type: 'scatter',
            symbol: 'diamond',
            symbolSize: 8,
            itemStyle: { color: COLORS.warningPoint },
            data: warningPoints,
          },
        ],
      },
      { notMerge: true },
    )
  }, [windows, threshold, episodes, warnings, reports])

  return (
    <figure className="flex flex-col gap-2">
      <div
        ref={ref}
        role="img"
        aria-label="Compressor duty score over time against the alarm threshold, with air-leak alarm episodes shaded red, sensor-consistency warnings marked in cyan, and operator failure reports shaded amber"
        className="h-72 w-full"
      />
      <figcaption className="flex flex-wrap gap-4 text-xs text-muted">
        <span>Blue line: primary duty score (gaps indicate abstained/invalid telemetry)</span>
        <span className="text-critical">Red band: air-leak alarm episode</span>
        <span>Cyan marker/band: sensor-consistency warning (not an air leak)</span>
        <span className="text-warn">Amber band: operator failure report</span>
      </figcaption>
    </figure>
  )
}
