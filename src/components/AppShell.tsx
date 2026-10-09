import { useState, type ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import '../App.css'

const navigation = [
  { label: 'Dashboard', path: '/', icon: 'dashboard' },
  { label: 'Live Transactions', path: '/transactions', icon: 'transactions' },
  { label: 'Network Explorer', path: '/network', icon: 'network' },
  { label: 'Investigation Cases', path: '/cases', icon: 'cases' },
  { label: 'Analytics', path: '/analytics', icon: 'analytics' },
] as const

function NavIcon({ name }: { name: (typeof navigation)[number]['icon'] }) {
  const common = {
    width: 17,
    height: 17,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.7,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true as const,
  }

  const paths = {
    dashboard: (
      <>
        <rect x="3.5" y="3.5" width="7" height="7" rx="1.2" />
        <rect x="13.5" y="3.5" width="7" height="7" rx="1.2" />
        <rect x="3.5" y="13.5" width="7" height="7" rx="1.2" />
        <rect x="13.5" y="13.5" width="7" height="7" rx="1.2" />
      </>
    ),
    transactions: (
      <>
        <path d="M4 6h16M4 12h16M4 18h10" />
        <circle cx="18" cy="18" r="2.5" />
      </>
    ),
    network: (
      <>
        <circle cx="6" cy="12" r="2.5" />
        <circle cx="18" cy="6" r="2.5" />
        <circle cx="18" cy="18" r="2.5" />
        <path d="m8.2 10.9 7.5-3.8M8.2 13.1l7.5 3.8" />
      </>
    ),
    cases: (
      <>
        <path d="M4 7.5h6l2 2h8v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
        <path d="M4 7.5v-1a2 2 0 0 1 2-2h4l2 2h6" />
      </>
    ),
    analytics: (
      <>
        <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
      </>
    ),
  }

  return (
    <span className="nav-icon">
      <svg {...common}>{paths[name]}</svg>
    </span>
  )
}

export function AppShell({ children }: { children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()
  const currentPage = navigation.find(({ path }) =>
    path === '/' ? location.pathname === '/' : location.pathname.startsWith(path),
  )?.label ?? 'Page'

  return (
    <div className="app-frame">
      <button
        aria-label="Close navigation"
        className={`overlay${menuOpen ? ' is-visible' : ''}`}
        onClick={() => setMenuOpen(false)}
        tabIndex={menuOpen ? 0 : -1}
        type="button"
      />
      <aside aria-label="Main navigation" className={`sidebar${menuOpen ? ' is-open' : ''}`}>
        <a aria-label="FinCrime Monitor home" className="brand" href="/">
          <span aria-hidden="true" className="brand-mark">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3 20 6v5c0 5-3.4 8.5-8 10-4.6-1.5-8-5-8-10V6z" />
              <path d="m8.5 12 2.2 2.2 4.8-5" />
            </svg>
          </span>
          <span className="brand-copy">
            <span className="brand-title">FINCRIME MONITOR</span>
            <span className="brand-subtitle">Financial intelligence</span>
          </span>
        </a>

        <p className="nav-label">Workspace</p>
        <nav aria-label="Workspace" className="primary-nav">
          {navigation.map(({ label, path, icon }) => (
            <NavLink
              key={path}
              aria-label={label}
              className="nav-link"
              end={path === '/'}
              onClick={() => setMenuOpen(false)}
              to={path}
            >
              <NavIcon name={icon} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <strong>Investigator workspace</strong>
          Evidence-led review · Data source not connected
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <button
            aria-expanded={menuOpen}
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            className="mobile-menu-button"
            onClick={() => setMenuOpen((open) => !open)}
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
              {menuOpen ? <path d="m6 6 12 12M18 6 6 18" /> : <path d="M4 6h16M4 12h16M4 18h16" />}
            </svg>
          </button>
          <div aria-live="polite" className="topbar-context">
            Workspace <span aria-hidden="true">/</span> <strong>{currentPage}</strong>
          </div>
          <div className="service-indicator">
            <span aria-hidden="true" className="service-dot" />
            Backend contract pending
          </div>
        </header>
        <main className="main-content">{children}</main>
      </div>
    </div>
  )
}
