import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { processAnalysisJob } from '../src/services/csv-analysis-jobs.js';
import type { CsvTransaction } from '../src/services/csv-analysis.js';

const mocks = vi.hoisted(() => ({
  poolQuery: vi.fn(),
  clientQuery: vi.fn(),
  connect: vi.fn(),
  assessTransaction: vi.fn()
}));

vi.mock('../src/config/database.js', () => ({
  pool: {
    query: mocks.poolQuery,
    connect: mocks.connect
  }
}));

vi.mock('../src/services/ai-client.js', () => ({
  assessTransaction: mocks.assessTransaction
}));

const savedAssessment = {
  status: 'available' as const,
  finalScore: 35,
  category: 'Monitor' as const,
  componentScores: {
    transaction_classifier: 30,
    anomaly_detector: 40,
    gnn_detector: 35
  },
  graphFindings: [],
  reasons: [],
  modelVersion: 'test'
};

const cycle: CsvTransaction[] = [
  { rowNumber: 2, id: 'AB', senderAccountId: 'A', receiverAccountId: 'B', amount: 100, timestamp: '2026-01-01T01:00:00.000Z', storageTimestamp: '2026-01-01T01:00:00Z', sourceTimestamp: '2026-01-01 01:00:00' },
  { rowNumber: 3, id: 'BC', senderAccountId: 'B', receiverAccountId: 'C', amount: 100, timestamp: '2026-01-01T02:00:00.000Z', storageTimestamp: '2026-01-01T02:00:00Z', sourceTimestamp: '2026-01-01 02:00:00' },
  { rowNumber: 4, id: 'CA', senderAccountId: 'C', receiverAccountId: 'A', amount: 100, timestamp: '2026-01-01T03:00:00.000Z', storageTimestamp: '2026-01-01T03:00:00Z', sourceTimestamp: '2026-01-01 03:00:00' }
];

function savedRow(transaction: CsvTransaction, matchesUpload: boolean) {
  return {
    id: transaction.id,
    sender_account_id: transaction.senderAccountId,
    receiver_account_id: transaction.receiverAccountId,
    amount: transaction.amount,
    occurred_at: transaction.storageTimestamp,
    channel: 'csv_import',
    risk_assessment: savedAssessment,
    matches_upload: matchesUpload
  };
}

async function analyzeSavedRows(matchesUpload: boolean[]) {
  mocks.poolQuery.mockImplementation(async (sql: string) => {
    if (sql.includes('JOIN jsonb_to_recordset')) {
      return { rows: cycle.map((transaction, index) => savedRow(transaction, matchesUpload[index]!)) };
    }
    return { rows: [] };
  });
  mocks.clientQuery.mockImplementation(async (sql: string) => (
    sql.includes('SELECT row_number, transaction_id, errors')
      ? { rows: [] }
      : { rows: [] }
  ));

  await processAnalysisJob('analysis-id', 'cycle.csv', cycle.length, cycle, []);
  const completion = mocks.clientQuery.mock.calls.find(([sql]) => sql.includes("status = 'succeeded'"));
  expect(completion).toBeDefined();
  return JSON.parse(completion![1][4] as string) as {
    summary: {
      acceptedRows: number;
      rejectedRows: number;
      nodeCount: number;
      edgeCount: number;
      patternCounts: Record<string, number>;
      scoredTransactionCount: number;
      unscoredTransactionCount: number;
    };
    graph: { edges: unknown[] };
    patterns: Array<{ pattern: string }>;
  };
}

describe('CSV analysis job duplicate handling', () => {
  beforeEach(() => {
    mocks.poolQuery.mockReset();
    mocks.clientQuery.mockReset();
    mocks.assessTransaction.mockReset();
    mocks.connect.mockReset().mockResolvedValue({
      query: mocks.clientQuery,
      release: vi.fn()
    });
  });

  afterEach(() => vi.restoreAllMocks());

  it('rebuilds graph findings from exact duplicate transfers without re-importing them', async () => {
    const result = await analyzeSavedRows([true, true, true]);

    expect(result.summary).toMatchObject({
      acceptedRows: 0,
      rejectedRows: 3,
      nodeCount: 3,
      edgeCount: 3,
      scoredTransactionCount: 3,
      unscoredTransactionCount: 0
    });
    expect(result.summary.patternCounts.cycle).toBe(1);
    expect(result.graph.edges).toHaveLength(3);
    expect(result.patterns.some((pattern) => pattern.pattern === 'cycle')).toBe(true);
    expect(mocks.assessTransaction).not.toHaveBeenCalled();
  });

  it('does not include duplicate IDs whose uploaded transfer details conflict with the saved record', async () => {
    const result = await analyzeSavedRows([false, false, false]);

    expect(result.graph.edges).toHaveLength(0);
    expect(result.patterns).toEqual([]);
  });
});
