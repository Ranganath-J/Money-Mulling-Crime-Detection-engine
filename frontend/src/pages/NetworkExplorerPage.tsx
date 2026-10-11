import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent } from 'react'
import { MetricCard } from '../components/MetricCard'
import { NetworkGraph } from '../components/NetworkGraph'
import { PageHeader } from '../components/PageHeader'
import { Panel } from '../components/Panel'
import { StateNotice } from '../components/StateNotice'
import { apiRequest } from '../lib/apiClient'
import type {
  AnalysisAccount,
  AnalysisGraphEdge,
  AnalysisGraphNode,
  AnalysisGraphResponse,
  AnalysisPatternFinding,
  AnalysisPatternType,
  CsvAnalysis,
} from '../types/api'

const patternLabels: Record<AnalysisPatternType, string> = {
  cycle: 'Circular routing',
  'fan-in': 'Fan-in',
  'fan-out': 'Fan-out',
  'layered-chain': 'Layered shell chain',
  'rapid-pass-through': 'Rapid pass-through',
  'high-velocity': 'High-velocity transfers',
}

type AnalysisListResponse = { analyses: CsvAnalysis[] }
type AnalysisResponse = { analysis: CsvAnalysis }
type PatternsResponse = { patterns: AnalysisPatternFinding[] }
type AccountsResponse = { accounts: AnalysisAccount[] }

function formatAmount(value: number): string {
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' }).format(value)
}

function riskLabel(value: number | null): string {
  if (value === null) return 'Not available'
  if (value >= 75) return `Investigation Alert · ${value.toFixed(1)}`
  if (value >= 40) return `Review · ${value.toFixed(1)}`
  return `Monitor · ${value.toFixed(1)}`
}

export function NetworkExplorerPage() {
  const [file, setFile] = useState<File | null>(null)
  const [analysis, setAnalysis] = useState<CsvAnalysis | null>(null)
  const [history, setHistory] = useState<CsvAnalysis[]>([])
  const [graph, setGraph] = useState<AnalysisGraphResponse | null>(null)
  const [patterns, setPatterns] = useState<AnalysisPatternFinding[]>([])
  const [accounts, setAccounts] = useState<AnalysisAccount[]>([])
  const [selectedNode, setSelectedNode] = useState<AnalysisGraphNode | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<AnalysisGraphEdge | null>(null)
  const [patternFilter, setPatternFilter] = useState('all')
  const [accountSearch, setAccountSearch] = useState('')
  const [minRisk, setMinRisk] = useState(0)
  const [maxRisk, setMaxRisk] = useState(100)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const fetchResults = useCallback(async (analysisId: string) => Promise.all([
      apiRequest<AnalysisGraphResponse>(`/analysis/${analysisId}/graph?limit=5000`),
      apiRequest<PatternsResponse>(`/analysis/${analysisId}/patterns`),
      apiRequest<AccountsResponse>(`/analysis/${analysisId}/accounts`),
    ]), [])

  const loadAnalysis = useCallback(async (analysisId: string) => {
    const response = await apiRequest<AnalysisResponse>(`/analysis/${analysisId}`)
    setAnalysis(response.analysis)
    setSelectedNode(null)
    setSelectedEdge(null)
  }, [])

  useEffect(() => {
    let active = true
    void apiRequest<AnalysisListResponse>('/analysis?limit=20')
      .then(async ({ analyses }) => {
        if (!active) return
        setHistory(analyses)
        setIsLoadingHistory(false)
        const latest = analyses[0]
        if (latest) {
          const response = await apiRequest<AnalysisResponse>(`/analysis/${latest.id}`)
          if (active) setAnalysis(response.analysis)
        }
      })
      .catch((cause: unknown) => {
        if (active) {
          setError(cause instanceof Error ? cause.message : 'Previous analyses could not be loaded.')
          setIsLoadingHistory(false)
        }
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (analysis?.status !== 'queued' && analysis?.status !== 'processing') return
    let active = true
    let timer = 0
    const poll = async () => {
      try {
        const response = await apiRequest<AnalysisResponse>(`/analysis/${analysis.id}`)
        if (!active) return
        setAnalysis(response.analysis)
        setHistory((items) => [response.analysis, ...items.filter((item) => item.id !== response.analysis.id)])
        if (response.analysis.status === 'queued' || response.analysis.status === 'processing') {
          timer = window.setTimeout(() => void poll(), 1200)
        }
      } catch (cause) {
        if (active) {
          setError(cause instanceof Error ? cause.message : 'Analysis status could not be refreshed.')
          timer = window.setTimeout(() => void poll(), 2500)
        }
      }
    }
    timer = window.setTimeout(() => void poll(), 700)
    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [analysis?.id, analysis?.status])

  useEffect(() => {
    if (analysis?.status !== 'succeeded') return
    let active = true
    void fetchResults(analysis.id)
      .then(([graphResponse, patternsResponse, accountsResponse]) => {
        if (!active) return
        setGraph(graphResponse)
        setPatterns(patternsResponse.patterns)
        setAccounts(accountsResponse.accounts)
        setError(null)
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : 'Analysis results could not be loaded.')
      })
    return () => {
      active = false
    }
  }, [analysis?.id, analysis?.status, fetchResults])

  const chooseFile = (candidate: File | undefined) => {
    setError(null)
    setNotice(null)
    if (!candidate) return
    if (!candidate.name.toLowerCase().endsWith('.csv')) {
      setError('Choose a file with the .csv extension.')
      return
    }
    if (candidate.size === 0) {
      setError('The selected file is empty.')
      return
    }
    if (candidate.size > 10 * 1024 * 1024) {
      setError('CSV files must be 10 MB or smaller.')
      return
    }
    setFile(candidate)
  }

  const uploadFile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!file) {
      setError('Choose a CSV file before starting an analysis.')
      return
    }
    setIsUploading(true)
    setError(null)
    setNotice(null)
    setGraph(null)
    setPatterns([])
    setAccounts([])
    const body = new FormData()
    body.append('file', file)
    try {
      const response = await apiRequest<AnalysisResponse>('/analysis/upload', { method: 'POST', body })
      setAnalysis(response.analysis)
      setHistory((items) => [response.analysis, ...items.filter((item) => item.id !== response.analysis.id)])
      setNotice(`Analysis ${response.analysis.id} was queued. Results will appear here when processing finishes.`)
      setFile(null)
      if (fileInput.current) fileInput.current.value = ''
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The CSV could not be uploaded.')
    } finally {
      setIsUploading(false)
    }
  }

  const selectHistoryItem = async (analysisId: string) => {
    if (!analysisId) return
    setError(null)
    setNotice(null)
    setGraph(null)
    setPatterns([])
    setAccounts([])
    try {
      await loadAnalysis(analysisId)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'That analysis could not be loaded.')
    }
  }

  const downloadReport = async () => {
    if (!analysis) return
    setError(null)
    try {
      const report = await apiRequest<unknown>(`/analysis/${analysis.id}/export`)
      const url = URL.createObjectURL(new Blob([`${JSON.stringify(report, null, 2)}\n`], { type: 'application/json' }))
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `analysis-${analysis.id}.json`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The analysis report could not be downloaded.')
    }
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    chooseFile(event.dataTransfer.files[0])
  }

  const visibleAccounts = useMemo(() => {
    const query = accountSearch.trim().toLowerCase()
    return accounts.filter((account) => !query || account.id.toLowerCase().includes(query))
  }, [accountSearch, accounts])

  const summary = analysis?.summary
  const thresholds = analysis?.thresholds ?? {}
  const fanWindowHours = thresholds.fanWindowHours ?? 72
  const passWindowHours = thresholds.passThroughWindowHours ?? 24
  const passTolerancePercent = (thresholds.passThroughAmountTolerance ?? 0.2) * 100
  const velocityWindowHours = thresholds.highVelocityWindowHours ?? 1
  const velocityCount = thresholds.highVelocityTransactionThreshold ?? 20
  const velocityAmount = thresholds.highVelocityAmountThreshold ?? 100_000
  const progress = analysis && analysis.totalRows > 0
    ? Math.min(100, Math.round((analysis.processedRows / analysis.totalRows) * 100))
    : analysis?.status === 'succeeded' ? 100 : 0

  return (
    <>
      <PageHeader
        title="CSV Network Analysis"
        description="Upload a transaction dataset to validate rows, score transactions with the available AI models, and review directed money-flow patterns."
      />
      {error && <StateNotice title="Analysis needs attention" description={error} variant="error" />}
      {notice && !error && <StateNotice title="Analysis started" description={notice} />}

      <Panel title="Upload transaction CSV" subtitle="UTF-8 CSV · up to 10 MB and 5,000 data rows · required columns may be reordered">
        <form className="csv-upload-form" onSubmit={uploadFile}>
          <div
            aria-label="Drop a CSV file here or choose one"
            className={`csv-dropzone${isDragging ? ' is-dragging' : ''}`}
            onDragEnter={(event) => { event.preventDefault(); setIsDragging(true) }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setIsDragging(false)
            }}
            onDrop={onDrop}
          >
            <input
              ref={fileInput}
              accept=".csv,text/csv"
              aria-label="Choose transaction CSV"
              className="csv-file-input"
              onChange={(event) => chooseFile(event.target.files?.[0])}
              type="file"
            />
            <div>
              <strong>{file ? file.name : 'Drop a transaction CSV here'}</strong>
              <span>{file ? `${(file.size / 1024).toFixed(1)} KB · ` : ''}or browse to select a file</span>
              <code>transaction_id, sender_id, receiver_id, amount, timestamp</code>
            </div>
            <button className="secondary-button" onClick={() => fileInput.current?.click()} type="button">
              Choose CSV
            </button>
          </div>
          <div className="csv-upload-actions">
            <p className="state-description">
              Valid rows are scored and added to the transaction ledger. Duplicate transaction IDs are rejected; invalid rows remain listed with their row numbers.
            </p>
            <button className="primary-button" disabled={isUploading || !file} type="submit">
              {isUploading ? 'Uploading…' : 'Upload and analyze'}
            </button>
          </div>
        </form>
      </Panel>

      <section aria-label="Previous analyses" className="analysis-history-bar">
        <label>
          Saved analyses
          <select
            aria-label="Select a saved analysis"
            disabled={isLoadingHistory || history.length === 0}
            onChange={(event) => void selectHistoryItem(event.target.value)}
            value={analysis?.id ?? ''}
          >
            {!history.length && <option value="">{isLoadingHistory ? 'Loading saved analyses…' : 'No saved analyses yet'}</option>}
            {history.map((item) => (
              <option key={item.id} value={item.id}>
                {item.fileName} · {item.status} · {item.createdAt ? new Date(item.createdAt).toLocaleString() : 'time unknown'}
              </option>
            ))}
          </select>
        </label>
        {analysis?.status === 'succeeded' && (
          <button className="secondary-button" onClick={() => void downloadReport()} type="button">
            Download documented JSON
          </button>
        )}
      </section>

      {analysis && (analysis.status === 'queued' || analysis.status === 'processing') && (
        <section aria-live="polite" className="panel analysis-progress">
          <div className="analysis-progress-copy">
            <strong>{analysis.status === 'queued' ? 'Queued for analysis' : 'Analyzing transaction dataset'}</strong>
            <span>{analysis.processedRows} of {analysis.totalRows} rows checked · {analysis.acceptedRows} accepted · {analysis.rejectedRows} rejected</span>
          </div>
          <progress aria-label="CSV analysis progress" max={100} value={progress} />
        </section>
      )}
      {analysis?.status === 'failed' && (
        <StateNotice title="Analysis failed" description={analysis.error ?? 'The analysis could not be completed.'} variant="error" />
      )}

      {analysis?.status === 'succeeded' && summary && (
        <>
          <StateNotice
            title={`${analysis.fileName} · analysis complete`}
            description={`${summary.acceptedRows} rows were accepted and ${summary.rejectedRows} rejected. Pattern findings are deterministic review signals, separate from AI component and fused risk scores; none is proof of wrongdoing.`}
          />
          {analysis.processingStats?.warnings.map((warning) => (
            <StateNotice key={warning} title="Some graph searches were bounded" description={warning} />
          ))}
          <div aria-label="Analysis row and pattern counts" className="metric-grid analysis-metric-grid">
            <MetricCard label="Accepted transfers" value={summary.acceptedRows} note="Valid rows persisted and assessed" />
            <MetricCard label="Rejected rows" marker="alert" value={summary.rejectedRows} note="Validation errors or duplicate IDs" />
            <MetricCard label="Circular routes" marker="alert" value={summary.patternCounts.cycle ?? 0} note="Cycles of 3–5 distinct accounts" />
            <MetricCard label="Fan-in findings" marker="review" value={summary.patternCounts['fan-in'] ?? 0} note={`Distinct senders within ${fanWindowHours} hours`} />
            <MetricCard label="Fan-out findings" marker="review" value={summary.patternCounts['fan-out'] ?? 0} note={`Distinct receivers within ${fanWindowHours} hours`} />
            <MetricCard label="Layered chains" marker="alert" value={summary.patternCounts['layered-chain'] ?? 0} note={`At least ${thresholds.layeredChainMinHops ?? 3} hops with timely amount matches`} />
            <MetricCard label="Rapid pass-through" marker="review" value={summary.patternCounts['rapid-pass-through'] ?? 0} note={`Forwarded within ${passWindowHours} hours · ±${passTolerancePercent}%`} />
            <MetricCard label="High velocity" marker="review" value={summary.patternCounts['high-velocity'] ?? 0} note={`${velocityCount}+ transfers or ${formatAmount(velocityAmount)} in ${velocityWindowHours} hour(s)`} />
          </div>

          <div className="analysis-filter-bar" aria-label="Network filters">
            <label>
              Find account
              <input onChange={(event) => setAccountSearch(event.target.value)} placeholder="Search account ID" value={accountSearch} />
            </label>
            <label>
              Pattern type
              <select onChange={(event) => setPatternFilter(event.target.value)} value={patternFilter}>
                <option value="all">All transfers</option>
                {Object.entries(patternLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label>
              Minimum AI risk: {minRisk}
              <input max="100" min="0" onChange={(event) => setMinRisk(Number(event.target.value))} type="range" value={minRisk} />
            </label>
            <label>
              Maximum AI risk: {maxRisk}
              <input max="100" min={minRisk} onChange={(event) => setMaxRisk(Number(event.target.value))} type="range" value={maxRisk} />
            </label>
            <button
              className="text-button"
              onClick={() => { setAccountSearch(''); setPatternFilter('all'); setMinRisk(0); setMaxRisk(100) }}
              type="button"
            >
              Clear filters
            </button>
          </div>

          <div className="analysis-results-grid">
            <Panel title="Directed transaction network" subtitle="Each arrow is an individual transfer; select accounts or transfers to inspect the underlying evidence.">
              {graph
                ? <NetworkGraph
                  accountSearch={accountSearch}
                  edges={graph.edges}
                  maxRisk={maxRisk}
                  minRisk={minRisk}
                  nodes={graph.nodes}
                  onSelectEdge={(edge) => { setSelectedEdge(edge); setSelectedNode(null) }}
                  onSelectNode={(node) => { setSelectedNode(node); setSelectedEdge(null) }}
                  patterns={patterns}
                  selectedPattern={patternFilter}
                />
                : <div className="panel-body"><p className="state-description">Loading graph data…</p></div>}
            </Panel>
            <Panel title="Selected evidence" subtitle="Choose an account or arrow in the network to see its details">
              <div className="panel-body analysis-selection">
                {selectedNode ? (
                  <>
                    <p className="page-eyebrow">Account</p>
                    <h3>{selectedNode.id}</h3>
                    <dl>
                      <dt>Transfers received / sent</dt><dd>{selectedNode.inDegree} / {selectedNode.outDegree}</dd>
                      <dt>Total incoming value</dt><dd>{formatAmount(selectedNode.incomingAmount)}</dd>
                      <dt>Total outgoing value</dt><dd>{formatAmount(selectedNode.outgoingAmount)}</dd>
                      <dt>Average incident AI risk</dt><dd>{riskLabel(selectedNode.riskScore)}</dd>
                      <dt>Pattern findings</dt><dd>{selectedNode.patternCount}</dd>
                    </dl>
                    <h4>AI component scores</h4>
                    {Object.keys(selectedNode.componentScores).length
                      ? <ul>{Object.entries(selectedNode.componentScores).map(([name, score]) =>                       <li key={name}>{name.replaceAll('_', ' ')}: {score == null ? 'not available' : `${score.toFixed(1)} / 100`}</li>)}</ul>
                      : <p className="state-description">No AI component score is available for this account.</p>}
                    <h4>Related findings</h4>
                    <ul>{patterns.filter((pattern) => pattern.accountIds.includes(selectedNode.id)).map((pattern) => <li key={pattern.id}>{pattern.description}</li>)}</ul>
                  </>
                ) : selectedEdge ? (
                  <>
                    <p className="page-eyebrow">Transfer</p>
                    <h3>{selectedEdge.transactionId}</h3>
                    <dl>
                      <dt>Route</dt><dd>{selectedEdge.source} → {selectedEdge.target}</dd>
                      <dt>Amount</dt><dd>{formatAmount(selectedEdge.amount)}</dd>
                      <dt>Source timestamp</dt><dd>{selectedEdge.sourceTimestamp}</dd>
                      <dt>Fused AI score</dt><dd>{riskLabel(selectedEdge.riskAssessment.finalScore ?? null)}</dd>
                      <dt>Assessment status</dt><dd>{selectedEdge.riskAssessment.status ?? 'unknown'}</dd>
                    </dl>
                    <h4>Model component scores</h4>
                    <ul>{Object.entries(selectedEdge.riskAssessment.componentScores ?? {}).map(([name, score]) => (
                      <li key={name}>{name.replaceAll('_', ' ')}: {score == null ? 'not available' : `${score.toFixed(1)} / 100`}</li>
                    ))}</ul>
                    <h4>AI-provided reasons</h4>
                    {selectedEdge.riskAssessment.reasons?.length
                      ? <ul>{selectedEdge.riskAssessment.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
                      : <p className="state-description">No model explanation was returned.</p>}
                    <h4>Deterministic findings</h4>
                    <ul>{patterns.filter((pattern) => selectedEdge.patternIds.includes(pattern.id)).map((pattern) => <li key={pattern.id}>{pattern.description}</li>)}</ul>
                  </>
                ) : (
                  <p className="state-description">No selection yet. Click an account node or a directed transfer in the graph.</p>
                )}
              </div>
            </Panel>
          </div>

          <div className="analysis-results-grid section-spacer">
            <Panel title="Accounts for investigator review" subtitle={`${visibleAccounts.length} accounts with graph findings or an AI risk score of 40+`}>
              {visibleAccounts.length ? (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead><tr><th>Account</th><th>Received / sent</th><th>Incoming / outgoing</th><th>AI risk</th><th>Findings</th></tr></thead>
                    <tbody>{visibleAccounts.slice(0, 200).map((account) => (
                      <tr key={account.id} onClick={() => { setSelectedNode(account); setSelectedEdge(null) }}>
                        <td><button className="table-link" onClick={() => { setSelectedNode(account); setSelectedEdge(null) }} type="button">{account.id}</button></td>
                        <td>{account.inDegree} / {account.outDegree}</td>
                        <td>{formatAmount(account.incomingAmount)} / {formatAmount(account.outgoingAmount)}</td>
                        <td>{riskLabel(account.riskScore)}</td>
                        <td>{account.patternCount}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              ) : <div className="panel-body"><p className="state-description">No accounts meet the current search or review criteria.</p></div>}
            </Panel>

            <Panel title="Detected graph patterns" subtitle={`${patterns.length} deterministic findings; select one to filter the network`}>
              {patterns.length ? (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead><tr><th>Pattern</th><th>Evidence summary</th><th>Accounts / transfers</th><th>Evidence</th></tr></thead>
                    <tbody>{patterns.filter((pattern) => patternFilter === 'all' || pattern.pattern === patternFilter).map((pattern) => (
                      <tr key={pattern.id}>
                        <td><button className="table-link" onClick={() => setPatternFilter(pattern.pattern)} type="button">{patternLabels[pattern.pattern]}</button></td>
                        <td>{pattern.description}</td>
                        <td>{pattern.accountIds.length} / {pattern.edgeIds.length}</td>
                        <td><details className="pattern-evidence">
                          <summary>View evidence</summary>
                          <pre>{JSON.stringify(pattern.evidence, null, 2)}</pre>
                          <p>Transfers: {pattern.edgeIds.join(', ')}</p>
                        </details></td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              ) : <div className="panel-body"><p className="state-description">No configured graph patterns were found in this dataset. This is not proof that activity is legitimate.</p></div>}
            </Panel>
          </div>

          {analysis.validationErrors?.length ? (
            <div className="section-spacer">
              <Panel title="Rejected rows" subtitle={`${analysis.validationErrors.length} row(s) were not added to the transaction ledger`}>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead><tr><th>CSV row</th><th>Transaction ID</th><th>Why it was rejected</th></tr></thead>
                    <tbody>{analysis.validationErrors.slice(0, 200).map((item) => (
                      <tr key={`${item.rowNumber}-${item.errors.join('|')}`}>
                        <td>{item.rowNumber}</td><td>{item.transactionId ?? '—'}</td><td>{item.errors.join(' ')}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
                {analysis.validationErrors.length > 200 && <p className="panel-body state-description">Showing the first 200 row errors. The JSON export contains every validation result.</p>}
              </Panel>
            </div>
          ) : null}

          {analysis.interpretation?.length ? (
            <div className="section-spacer">
              <StateNotice title="How to interpret this report" description={analysis.interpretation.join(' ')} />
            </div>
          ) : null}
        </>
      )}
    </>
  )
}
