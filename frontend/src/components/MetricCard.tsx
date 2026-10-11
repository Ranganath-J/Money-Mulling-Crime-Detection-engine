type MetricCardProps = {
  label: string
  marker?: 'monitor' | 'review' | 'alert'
  value?: string | number
  note?: string
}

export function MetricCard({ label, marker = 'monitor', value, note = 'Awaiting backend data' }: MetricCardProps) {
  return (
    <section aria-label={`${label}: ${value ?? 'unavailable'}`} className="metric-card">
      <div className="metric-label">
        <span aria-hidden="true" className={`metric-marker ${marker}`} />
        {label}
      </div>
      <div className="metric-value">{value ?? 'Unavailable'}</div>
      <div className="metric-note">{note}</div>
    </section>
  )
}
