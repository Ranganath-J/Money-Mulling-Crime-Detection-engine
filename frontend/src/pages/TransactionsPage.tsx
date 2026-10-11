import { useEffect, useState, type FormEvent } from 'react'
import { PageHeader } from '../components/PageHeader'
import { Panel } from '../components/Panel'
import { StateNotice } from '../components/StateNotice'
import { TransactionTable } from '../components/TransactionTable'
import { apiRequest } from '../lib/apiClient'
import type { Transaction } from '../types/api'

type TransactionResponse = {
  transaction: Transaction
  warnings: string[]
}

export function TransactionsPage() {
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [senderAccountId, setSenderAccountId] = useState('')
  const [receiverAccountId, setReceiverAccountId] = useState('')
  const [amount, setAmount] = useState('')
  const [currency, setCurrency] = useState('USD')
  const [channel, setChannel] = useState('online')

  useEffect(() => {
    let active = true
    void apiRequest<{ transactions: Transaction[] }>('/transactions?limit=100')
      .then((result) => {
        if (active) {
          setTransactions(result.transactions)
          setLoadError(null)
        }
      })
      .catch((cause: unknown) => {
        if (active) setLoadError(cause instanceof Error ? cause.message : 'Could not load transactions.')
      })
    return () => {
      active = false
    }
  }, [refreshKey])

  const submitTransaction = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setIsSubmitting(true)
    setSubmitError(null)
    setMessage(null)
    try {
      const result = await apiRequest<TransactionResponse>('/transactions', {
        method: 'POST',
        body: JSON.stringify({
          senderAccountId,
          receiverAccountId,
          amount: Number(amount),
          currency,
          channel
        })
      })
      const riskAssessment = result.transaction.riskAssessment
      setMessage(
        riskAssessment?.finalScore == null
          ? `Transaction ${result.transaction.id} saved, but no model score was available.`
          : `Transaction ${result.transaction.id} scored ${riskAssessment.finalScore.toFixed(1)} (${riskAssessment.category}).`
      )
      if (result.warnings.length) setSubmitError(`Model warning: ${result.warnings.join(' ')}`)
      setSenderAccountId('')
      setReceiverAccountId('')
      setAmount('')
      setRefreshKey((key) => key + 1)
    } catch (cause) {
      setSubmitError(cause instanceof Error ? cause.message : 'Transaction could not be submitted.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <>
      <PageHeader
        title="Live Transactions"
        description="Submit a transaction to store it, score it with the AI models, and refresh the dashboard."
      />

      {submitError && <StateNotice title="Transaction processing needs attention" description={submitError} variant="error" />}
      {message && !submitError && <StateNotice title="Transaction processed" description={message} />}
      {loadError && <StateNotice title="Transactions could not be loaded" description={loadError} variant="error" />}

      <Panel title="Submit a transaction" subtitle="Amounts are capped at 10,000,000 per the AI service contract">
        <form className="transaction-form" onSubmit={submitTransaction}>
          <label>
            Sender account
            <input autoComplete="off" maxLength={120} onChange={(event) => setSenderAccountId(event.target.value)} required value={senderAccountId} />
          </label>
          <label>
            Receiver account
            <input autoComplete="off" maxLength={120} onChange={(event) => setReceiverAccountId(event.target.value)} required value={receiverAccountId} />
          </label>
          <label>
            Amount
            <input max="10000000" min="0.01" onChange={(event) => setAmount(event.target.value)} required step="0.01" type="number" value={amount} />
          </label>
          <label>
            Currency
            <input maxLength={3} minLength={3} onChange={(event) => setCurrency(event.target.value.toUpperCase())} pattern="[A-Za-z]{3}" required title="Enter a three-letter currency code" value={currency} />
          </label>
          <label>
            Channel
            <select onChange={(event) => setChannel(event.target.value)} value={channel}>
              <option value="online">Online</option>
              <option value="mobile">Mobile</option>
              <option value="wire">Wire</option>
              <option value="atm">ATM</option>
              <option value="pos">Point of sale</option>
            </select>
          </label>
          <button className="primary-button" disabled={isSubmitting} type="submit">
            {isSubmitting ? 'Scoring transaction…' : 'Submit and score'}
          </button>
        </form>
      </Panel>

      <div className="section-spacer">
        <Panel title="Transaction history" subtitle="Persisted records and model output">
          {loadError
            ? <div className="panel-body"><p className="state-description">Connect PostgreSQL and the backend to view saved transactions.</p></div>
            : <TransactionTable transactions={transactions} />}
        </Panel>
      </div>
    </>
  )
}
