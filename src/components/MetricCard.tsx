type MetricCardProps = {
  label: string
  marker?: 'monitor' | 'review' | 'alert'
  note?: string
}

export function MetricCard({ label, marker = 'monitor', note = 'Awaiting backend data' }: MetricCardProps) {
  return (
    <section aria-label={`${label}: unavailable`} className="metric-card">
      <div className="metric-label">
        <span aria-hidden="true" className={`metric-marker ${marker}`} />
        {label}
      </div>
      <div className="metric-value">Unavailable</div>
      <div className="metric-note">{note}</div>
    </section>
  )
}
