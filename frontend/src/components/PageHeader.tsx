type PageHeaderProps = {
  title: string
  description: string
}

export function PageHeader({ title, description }: PageHeaderProps) {
  return (
    <div className="page-header">
      <div>
        <p className="page-eyebrow">Financial intelligence</p>
        <h1 className="page-title">{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      <div className="contract-badge">
        <span aria-hidden="true" className="service-dot" />
        Live service health in top bar
      </div>
    </div>
  )
}
