import type { ReactNode } from 'react'

type PanelProps = {
  title: string
  subtitle?: string
  children: ReactNode
}

export function Panel({ title, subtitle, children }: PanelProps) {
  return (
    <section aria-label={title} className="panel">
      <header className="panel-header">
        <div>
          <h2 className="panel-title">{title}</h2>
          {subtitle && <p className="panel-subtitle">{subtitle}</p>}
        </div>
      </header>
      {children}
    </section>
  )
}
