import type { Transaction } from '../types/api'

type TransactionTableProps = {
  transactions: Transaction[]
}

const componentLabels: Record<string, string> = {
  transaction_classifier: 'Transaction behavior model (40% weight)',
  anomaly_detector: 'Account behavior model (20% weight)',
  gnn_detector: 'Network structure model (40% weight)'
}

const findingLabels: Record<string, string> = {
  cycle: 'Circular transfer pattern',
  'fan-in': 'Many accounts paying into one account',
  'fan-out': 'One account paying many accounts',
  'rapid-pass-through': 'Funds forwarded shortly after receipt',
  'layered-chain': 'Multi-step transfer chain'
}

const riskDescriptions: Record<string, string> = {
  Monitor: 'Lower model-indicated risk. Continue routine monitoring.',
  Review: 'Elevated model-indicated risk. Review the available evidence.',
  'Investigation Alert': 'Higher model-indicated risk. Prioritize investigator review.'
}

function formatAmount(amount: number, currency = 'USD') {
  return new Intl.NumberFormat(undefined, { style: 'currency', currency }).format(amount)
}

function formatFinding(finding: string) {
  return findingLabels[finding] ?? finding.replaceAll('-', ' ')
}

export function TransactionTable({ transactions }: TransactionTableProps) {
  if (!transactions.length) {
    return <div className="panel-body"><p className="state-description">No transactions have been submitted yet.</p></div>
  }

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Transaction ID</th>
            <th scope="col">Accounts</th>
            <th scope="col">Amount</th>
            <th scope="col">Investigation priority</th>
            <th scope="col">Model risk score</th>
            <th scope="col">Transaction time</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((transaction) => {
            const assessment = transaction.riskAssessment
            const category = assessment?.category
            return (
              <tr key={transaction.id}>
                <td>
                  <strong>{transaction.id}</strong>
                  <span className="table-secondary">{transaction.channel}</span>
                </td>
                <td>
                  <span>{transaction.senderAccountId}</span>
                  <span className="table-secondary">to {transaction.receiverAccountId}</span>
                </td>
                <td>{formatAmount(transaction.amount, transaction.currency)}</td>
                <td>
                  {category
                    ? (
                      <>
                        <span className={`risk-pill ${category === 'Investigation Alert' ? 'risk-alert' : category === 'Review' ? 'risk-review' : 'risk-monitor'}`}>{category}</span>
                        <span className="table-secondary">{riskDescriptions[category] ?? 'Review the available evidence.'}</span>
                      </>
                    )
                    : <span className="risk-pill risk-unavailable">Unavailable</span>}
                </td>
                <td>
                  {assessment?.finalScore == null
                    ? <span aria-label="No usable model risk score">No score</span>
                    : <span>{assessment.finalScore.toFixed(1)} / 100</span>}
                  {assessment && (
                    <details className="assessment-details">
                      <summary>View assessment details</summary>
                      <div className="assessment-detail-content">
                        <p className="assessment-detail-heading">Assessment status: {assessment.status ?? 'not provided'}</p>
                        <p className="assessment-detail-heading">AI model signals</p>
                        <p className="assessment-detail-note">Standard score weights: transaction behavior 40%, account behavior 20%, network structure 40%. If a model has no usable score, the other available components are reweighted; missing scores are not treated as zero.</p>
                        {assessment.componentScores && (
                          <ul className="assessment-component-list">
                            {Object.entries(assessment.componentScores).map(([name, score]) => (
                              <li key={name}>
                                <span title={name}>{componentLabels[name] ?? name.replaceAll('_', ' ')}</span>
                                <strong>{score == null ? 'Unavailable' : `${score.toFixed(1)} / 100`}</strong>
                              </li>
                            ))}
                          </ul>
                        )}
                        {assessment.graphFindings && assessment.graphFindings.length > 0 && (
                          <>
                            <p className="assessment-detail-heading">Network findings (separate evidence)</p>
                            <ul className="assessment-reason-list">
                              {assessment.graphFindings.map((finding) => <li key={finding}>{formatFinding(finding)}</li>)}
                            </ul>
                          </>
                        )}
                        {assessment.reasons && assessment.reasons.length > 0 && (
                          <>
                            <p className="assessment-detail-heading">Model-provided reasons</p>
                            <ul className="assessment-reason-list">
                              {assessment.reasons.map((reason) => <li key={reason}>{reason}</li>)}
                            </ul>
                          </>
                        )}
                        <p className="assessment-detail-note">A high score indicates a review priority, not proof of wrongdoing. Confirm findings against source records and applicable procedures.</p>
                      </div>
                    </details>
                  )}
                </td>
                <td>
                  <span>{new Date(transaction.timestamp).toLocaleString()}</span>
                  <span className="table-secondary">{transaction.channel ?? 'Channel not recorded'}</span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
