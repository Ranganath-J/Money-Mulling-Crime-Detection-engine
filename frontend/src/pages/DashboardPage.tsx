import { useEffect, useState } from 'react'
import { MetricCard } from '../components/MetricCard'
import { PageHeader } from '../components/PageHeader'
import { Panel } from '../components/Panel'
import { StateNotice } from '../components/StateNotice'
import { TransactionTable } from '../components/TransactionTable'
import { apiRequest } from '../lib/apiClient'
import type { DashboardSummary } from '../types/api'

export function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const refresh = async () => {
      try {
        const data = await apiRequest<DashboardSummary>('/dashboard')
        if (active) {
          setSummary(data)
          setError(null)
        }
      } catch (cause) {
        if (active) setError(cause instanceof Error ? cause.message : 'Could not load dashboard data.')
      }
    }
    void refresh()
    const interval = window.setInterval(() => void refresh(), 10_000)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [])

  return (
    <>
      <PageHeader
        title="Operations overview"
        description="Monitor submitted transactions and risk scores returned by the detection models."
      />

      {error && <StateNotice title="Dashboard data could not be loaded" description={error} variant="error" />}
      <StateNotice
        title="How to read these risk indicators"
        description="Risk scores are model-generated investigation priorities, not proof of wrongdoing. Monitor: below 40. Review: 40–74.9. Investigation Alert: 75 or higher. Category counts exclude transactions without a usable score. Network findings are separate evidence, not an extra score. Open “View assessment details” in a transaction row to review its model signals, network findings, score calculation, and reasons."
      />

      <div aria-label="Transaction and risk summary" className="metric-grid">
        <MetricCard label="Transactions recorded" value={summary?.totalTransactions} note="Saved transactions, including those without a score" />
        <MetricCard label="Monitor" marker="monitor" value={summary?.monitorCount} note="Score below 40 · scored transactions" />
        <MetricCard label="Review" marker="review" value={summary?.reviewCount} note="Score 40–74.9 · scored transactions" />
        <MetricCard label="Investigation Alert" marker="alert" value={summary?.alertCount} note="Score 75+ · scored transactions" />
      </div>

      <Panel title="Recent transaction assessments" subtitle="Latest records and investigator-facing explanations of AI risk signals">
        {summary
          ? <TransactionTable transactions={summary.recentTransactions} />
          : !error && <div className="panel-body"><p className="state-description">Loading transactions…</p></div>}
      </Panel>
    </>
  )
}
