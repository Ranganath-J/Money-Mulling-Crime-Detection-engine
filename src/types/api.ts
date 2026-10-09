export type RiskCategory = 'Monitor' | 'Review' | 'Investigation Alert'

export interface ComponentScores {
  [componentName: string]: number | null | undefined
}

export interface RiskAssessment {
  finalScore?: number | null
  category?: RiskCategory | string
  componentScores?: ComponentScores
  reasons?: string[]
  modelVersion?: string
  ruleVersion?: string
}

export interface Transaction {
  id: string
  senderAccountId: string
  receiverAccountId: string
  amount: number
  currency?: string
  timestamp: string
  channel?: string
  riskAssessment?: RiskAssessment
  caseId?: string | null
  status?: string
}

export interface GraphNode {
  id: string
  label?: string
  attributes?: Record<string, unknown>
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  transactionId?: string
  amount?: number
  timestamp?: string
  attributes?: Record<string, unknown>
}

export type DetectionPattern =
  | 'cycle'
  | 'fan-in'
  | 'fan-out'
  | 'layered-chain'
  | 'rapid-pass-through'
  | string

export interface Detection {
  id: string
  pattern: DetectionPattern
  description?: string
  evidence?: Record<string, unknown>
  nodeIds?: string[]
  edgeIds?: string[]
}

export interface AccountNetwork {
  nodes: GraphNode[]
  edges: GraphEdge[]
  detections?: Detection[]
}

export interface InvestigationCase {
  id: string
  createdAt: string
  status: string
  riskAssessment?: RiskAssessment
  involvedAccountIds?: string[]
  topReasons?: string[]
  transactions?: Transaction[]
  notes?: string[]
}
