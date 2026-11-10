import { LineChart } from 'echarts/charts'
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
import type { AlarmEpisode, OperatorReport, WindowResult } from '../../api/generated/model'
import { formatNaive, toEpochMs } from '../../lib/naiveTime'

echarts.use([
  LineChart,
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
}

const COLORS = {
  line: '#3b82f6',
  threshold: '#f59e0b',
  alarm: 'rgba(239, 68, 68, 0.18)',
  report: 'rgba(245, 158, 11, 0.14)',
  axis: '#8a96ad',
  grid: '#1f2a40',
}

/** Anomaly score per window vs threshold, with alarm episodes and operator reports shaded.
 * Missing windows are left as gaps (never interpolated). */
export function ScoreChart({ windows, threshold, episodes, reports }: ScoreChartProps) {
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
          valueFormatter: (v: unknown) => (typeof v === 'number' ? v.toFixed(3) : 'no data'),
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
          name: 'anomaly score',
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
        ],
      },
      { notMerge: true },
    )
  }, [windows, threshold, episodes, reports])

  return (
    <div
      ref={ref}
      role="img"
      aria-label="Anomaly score over time against the alarm threshold, with alarm episodes in red and operator failure reports in amber"
      className="h-72 w-full"
    />
  )
}
