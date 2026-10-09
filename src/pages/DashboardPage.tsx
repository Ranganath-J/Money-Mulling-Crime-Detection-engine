import { MetricCard } from '../components/MetricCard'
import { PageHeader } from '../components/PageHeader'
import { Panel } from '../components/Panel'
import { StateNotice } from '../components/StateNotice'

function UnavailableChart({ title }: { title: string }) {
  return (
    <div className="chart-placeholder">
      <div className="chart-placeholder-inner">
        <span aria-hidden="true" className="chart-placeholder-icon">
          <svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 19V5m0 14h18M7 15l4-4 3 3 6-7" />
          </svg>
        </span>
        <h3 className="state-title">{title} unavailable</h3>
        <p className="state-description">This visualization will display measured backend data when the API contract is connected.</p>
      </div>
    </div>
  )
}

export function DashboardPage() {
  return (
    <>
      <PageHeader
        title="Operations overview"
        description="Monitor transaction activity and review system-confirmed risk signals."
      />

      <StateNotice
        title="Dashboard data is not connected"
        description="No live or illustrative values are shown. Connect the agreed backend API contract to load actual transaction counts, detections, model availability, and measured performance."
      />

      <div aria-label="Transaction and risk summary" className="metric-grid" style={{ marginTop: 16 }}>
        <MetricCard label="Transactions processed" />
        <MetricCard label="Monitor" marker="monitor" />
        <MetricCard label="Review" marker="review" />
        <MetricCard label="Investigation Alert" marker="alert" />
      </div>

      <div className="dashboard-grid">
        <Panel title="Transaction activity" subtitle="Processed activity over time">
          <UnavailableChart title="Transaction activity" />
        </Panel>
        <div className="panel-stack">
          <Panel title="Risk distribution" subtitle="Backend-classified risk categories">
            <div className="panel-body">
              <UnavailableChart title="Risk distribution" />
            </div>
          </Panel>
          <Panel title="Recent suspicious activity" subtitle="Items surfaced by backend detections">
            <div className="panel-body">
              <p className="state-title">No backend data available</p>
              <p className="state-description">Recent activity will appear here after the detection and transaction APIs are connected.</p>
            </div>
          </Panel>
          <Panel title="Model and service status" subtitle="Availability and measured performance">
            <div className="panel-body">
              <p className="state-description">Model availability, detection latency, and model metrics are not shown until measured values are provided by the backend.</p>
              <div className="metric-detail-grid">
                <div className="detail-tile">
                  <span className="detail-tile-label">AI / model availability</span>
                  <span className="detail-tile-value">Unavailable</span>
                </div>
                <div className="detail-tile">
                  <span className="detail-tile-label">Detection latency</span>
                  <span className="detail-tile-value">Unavailable</span>
                </div>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </>
  )
}
