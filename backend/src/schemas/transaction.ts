import { z } from 'zod';

export const transactionInputSchema = z.object({
  id: z.string().trim().min(1).max(120).optional(),
  senderAccountId: z.string().trim().min(1).max(120),
  receiverAccountId: z.string().trim().min(1).max(120),
  amount: z.number().finite().positive().max(10_000_000)
    .refine((value) => Number(value.toFixed(2)) === value, 'amount can have at most two decimal places'),
  currency: z.string().trim().toUpperCase().regex(/^[A-Z]{3}$/).default('USD'),
  timestamp: z.string().datetime({ offset: true }).optional(),
  channel: z.string().trim().min(1).max(40).default('online')
});

export type TransactionInput = z.infer<typeof transactionInputSchema>;

export type TransactionRecord = {
  id: string;
  senderAccountId: string;
  receiverAccountId: string;
  amount: number;
  currency: string;
  timestamp: string;
  channel: string;
  riskAssessment: RiskAssessment;
};

export type RiskAssessment = {
  status: 'available' | 'partial' | 'unavailable';
  finalScore: number | null;
  category: 'Monitor' | 'Review' | 'Investigation Alert' | null;
  componentScores: Record<string, number | null>;
  graphFindings: string[];
  reasons: string[];
  modelVersion: string | null;
};

export type DashboardSummary = {
  totalTransactions: number;
  monitorCount: number;
  reviewCount: number;
  alertCount: number;
  recentTransactions: TransactionRecord[];
};

export function rowToTransaction(row: Record<string, unknown>): TransactionRecord {
  const assessment = typeof row.risk_assessment === 'string'
    ? JSON.parse(row.risk_assessment) as RiskAssessment
    : row.risk_assessment as RiskAssessment;

  return {
    id: String(row.id),
    senderAccountId: String(row.sender_account_id),
    receiverAccountId: String(row.receiver_account_id),
    amount: Number(row.amount),
    currency: String(row.currency),
    timestamp: new Date(String(row.occurred_at)).toISOString(),
    channel: String(row.channel),
    riskAssessment: assessment
  };
}
