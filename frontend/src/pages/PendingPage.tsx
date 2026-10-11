import { PageHeader } from '../components/PageHeader'

type PendingPageProps = {
  title: string
  description: string
}

export function PendingPage({ title, description }: PendingPageProps) {
  return (
    <>
      <PageHeader title={title} description={description} />
      <section aria-label={`${title} data status`} className="panel">
        <div className="pending-content">
          <div className="pending-content-inner">
            <span aria-hidden="true" className="notice-icon">
              <svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="9" />
                <path d="M12 11v5m0-8h.01" />
              </svg>
            </span>
            <h2 className="state-title">Waiting for the agreed backend contract</h2>
            <p className="state-description">
              This view is intentionally not connected yet. Endpoint paths, response fields, event names, and permitted actions must come from the application owners before data or controls are added.
            </p>
          </div>
        </div>
      </section>
    </>
  )
}
