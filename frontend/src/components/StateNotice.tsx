type StateNoticeProps = {
  title: string
  description: string
  variant?: 'info' | 'error'
}

export function StateNotice({ title, description, variant = 'info' }: StateNoticeProps) {
  return (
    <div aria-live="polite" className={`notice notice-${variant}`} role={variant === 'error' ? 'alert' : 'status'}>
      <span aria-hidden="true" className="notice-icon">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
          {variant === 'error' ? (
            <>
              <circle cx="12" cy="12" r="9" />
              <path d="M12 7v5m0 4h.01" />
            </>
          ) : (
            <>
              <circle cx="12" cy="12" r="9" />
              <path d="M12 11v5m0-8h.01" />
            </>
          )}
        </svg>
      </span>
      <div className="notice-copy">
        <h2 className="notice-title">{title}</h2>
        <p className="notice-description">{description}</p>
      </div>
    </div>
  )
}
