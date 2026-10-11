export type RiskCategory = 'Monitor' | 'Review' | 'Investigation Alert'

export interface ComponentScores {
  [componentName: string]: number | null | undefined
}

export interface RiskAssessment {
  status?: 'available' | 'partial' | 'unavailable'
  finalScore?: number | null
  category?: RiskCategory | string
  componentScores?: ComponentScores
  graphFindings?: string[]
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

export interface DashboardSummary {
  totalTransactions: number
  monitorCount: number
  reviewCount: number
  alertCount: number
  recentTransactions: Transaction[]
}

export interface ServiceStatus {
  backend: string
  database: string
  aiService: string
  models?: Record<string, { status?: string }>
}

export type AnalysisStatus = 'queued' | 'processing' | 'succeeded' | 'failed'

export interface CsvAnalysisSummary {
  totalRows: number
  acceptedRows: number
  rejectedRows: number
  nodeCount: number
  edgeCount: number
  patternCounts: Record<string, number>
  scoredTransactionCount: number
  unscoredTransactionCount: number
}

export interface CsvValidationError {
  rowNumber: number
  transactionId?: string
  errors: string[]
}

export interface CsvAnalysis {
  id: string
  fileName: string
  status: AnalysisStatus
  totalRows: number
  processedRows: number
  acceptedRows: number
  rejectedRows: number
  summary: CsvAnalysisSummary | null
  thresholds?: Record<string, number>
  processingStats?: {
    rowsProcessed: number
    aiModelErrors: number
    warnings: string[]
  } | null
  validationErrors?: CsvValidationError[]
  interpretation?: string[]
  error?: string | null
  createdAt?: string
  updatedAt?: string
}

export type AnalysisPatternType =
  | 'cycle'
  | 'fan-in'
  | 'fan-out'
  | 'layered-chain'
  | 'rapid-pass-through'
  | 'high-velocity'

export interface AnalysisPatternFinding {
  id: string
  pattern: AnalysisPatternType
  severity: 'medium' | 'high'
  description: string
  accountIds: string[]
  edgeIds: string[]
  evidence: Record<string, unknown>
}

export interface AnalysisGraphNode {
  id: string
  inDegree: number
  outDegree: number
  transactionCount: number
  incomingAmount: number
  outgoingAmount: number
  patternCount: number
  riskScore: number | null
  componentScores: ComponentScores
  patternIds?: string[]
}

export interface AnalysisGraphEdge {
  id: string
  source: string
  target: string
  transactionId: string
  amount: number
  timestamp: string
  sourceTimestamp: string
  riskAssessment: RiskAssessment
  patternIds: string[]
}

export interface AnalysisGraphResponse {
  nodes: AnalysisGraphNode[]
  edges: AnalysisGraphEdge[]
  totalNodes: number
  totalEdges: number
  limit: number
  offset: number
}

export interface AnalysisAccount extends AnalysisGraphNode {
  patternIds: string[]
}
