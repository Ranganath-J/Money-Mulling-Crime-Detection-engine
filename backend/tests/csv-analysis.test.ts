import { describe, expect, it } from 'vitest';
import { analyzeGraph, CsvInputError, parseCsv, type ScoredCsvTransaction } from '../src/services/csv-analysis.js';

const assessment = {
  status: 'available' as const,
  finalScore: 20,
  category: 'Monitor' as const,
  componentScores: {
    transaction_classifier: 10,
    anomaly_detector: 20,
    gnn_detector: 30
  },
  graphFindings: [],
  reasons: [],
  modelVersion: 'test'
};

function transaction(
  id: string,
  senderAccountId: string,
  receiverAccountId: string,
  hour: number,
  amount = 100
): ScoredCsvTransaction {
  return {
    rowNumber: hour + 2,
    id,
    senderAccountId,
    receiverAccountId,
    amount,
    timestamp: new Date(Date.UTC(2026, 0, 1, hour)).toISOString(),
    storageTimestamp: new Date(Date.UTC(2026, 0, 1, hour)).toISOString(),
    sourceTimestamp: `2026-01-01 ${String(hour).padStart(2, '0')}:00:00`,
    assessment
  };
}

function patterns(items: ScoredCsvTransaction[]) {
  return analyzeGraph(items).patterns;
}

describe('CSV parsing and validation', () => {
  it('accepts a valid CSV with reordered required columns and preserves the source timestamp', () => {
    const result = parseCsv(Buffer.from([
      'timestamp,receiver_id,amount,transaction_id,sender_id',
      '2026-02-18 10:00:00,ACC_B,5000.25,TXN001,ACC_A'
    ].join('\r\n')));

    expect(result.totalRows).toBe(1);
    expect(result.validationErrors).toEqual([]);
    expect(result.transactions[0]).toMatchObject({
      id: 'TXN001',
      senderAccountId: 'ACC_A',
      receiverAccountId: 'ACC_B',
      amount: 5000.25,
      timestamp: '2026-02-18T10:00:00.000Z',
      storageTimestamp: '2026-02-18T10:00:00Z',
      sourceTimestamp: '2026-02-18 10:00:00'
    });
  });

  it('preserves sub-millisecond precision and timezone offsets for database storage', () => {
    const result = parseCsv(Buffer.from([
      'transaction_id,sender_id,receiver_id,amount,timestamp',
      'TX-FRACTION,ACC_A,ACC_B,5,2026-02-18T10:00:00.123456+01:00'
    ].join('\n')));
    expect(result.transactions[0]).toMatchObject({
      timestamp: '2026-02-18T09:00:00.123Z',
      storageTimestamp: '2026-02-18T10:00:00.123456+01:00',
      sourceTimestamp: '2026-02-18T10:00:00.123456+01:00'
    });
  });

  it('reports invalid amounts, invalid timestamps, and duplicate IDs with data row numbers', () => {
    const result = parseCsv(Buffer.from([
      'transaction_id,sender_id,receiver_id,amount,timestamp',
      'TX-1,ACC_A,ACC_B,-1,2026-02-18 10:00:00',
      'TX-2,ACC_A,ACC_B,12.345,not-a-date',
      'TX-1,ACC_A,ACC_C,5,2026-02-18 10:01:00'
    ].join('\n')));

    expect(result.transactions).toHaveLength(0);
    expect(result.validationErrors).toHaveLength(3);
    expect(result.validationErrors.map((item) => item.rowNumber)).toEqual([2, 3, 4]);
    expect(result.validationErrors[0]?.errors.join(' ')).toContain('positive number');
    expect(result.validationErrors[1]?.errors.join(' ')).toContain('Timestamp');
    expect(result.validationErrors[2]?.errors.join(' ')).toContain('Duplicate transaction ID');
  });

  it('rejects missing headers, empty files, malformed CSV, and non-UTF-8 input', () => {
    expect(() => parseCsv(Buffer.from('transaction_id,sender_id\nTX-1,ACC_A'))).toThrowError(
      expect.objectContaining({ code: 'MISSING_HEADERS' })
    );
    expect(() => parseCsv(Buffer.alloc(0))).toThrowError(expect.objectContaining({ code: 'EMPTY_FILE' }));
    expect(() => parseCsv(Buffer.from('transaction_id,sender_id,receiver_id,amount,timestamp\n"TX-1,ACC_A,ACC_B,5,2026-02-18 10:00:00')))
      .toThrowError(expect.objectContaining({ code: 'MALFORMED_CSV' }));
    expect(() => parseCsv(Buffer.from([0xff, 0xfe, 0xfd]))).toThrowError(
      expect.objectContaining({ code: 'UNSUPPORTED_ENCODING' })
    );
  });

  it('rejects unsupported file sizes with an actionable input error', () => {
    try {
      parseCsv(Buffer.alloc(10 * 1024 * 1024 + 1));
      throw new Error('Expected file size validation to fail');
    } catch (error) {
      expect(error).toBeInstanceOf(CsvInputError);
      expect((error as CsvInputError).code).toBe('FILE_TOO_LARGE');
    }
  });
});

describe('deterministic transaction graph findings', () => {
  it('finds a three-account directed cycle once', () => {
    const result = analyzeGraph([
      transaction('ab', 'A', 'B', 1),
      transaction('bc', 'B', 'C', 2),
      transaction('ca', 'C', 'A', 3)
    ]);
    const cycles = result.patterns.filter((pattern) => pattern.pattern === 'cycle');
    expect(cycles).toHaveLength(1);
    expect(cycles[0]?.accountIds).toEqual(['A', 'B', 'C']);
    expect(cycles[0]?.edgeIds).toHaveLength(3);
  });

  it('finds a five-account cycle but does not report a six-account-only cycle', () => {
    const five = ['A', 'B', 'C', 'D', 'E'];
    const fiveEdges = five.map((account, index) =>
      transaction(`five-${index}`, account, five[(index + 1) % five.length]!, index)
    );
    expect(patterns(fiveEdges).filter((pattern) => pattern.pattern === 'cycle')).toHaveLength(1);

    const six = ['A', 'B', 'C', 'D', 'E', 'F'];
    const sixEdges = six.map((account, index) =>
      transaction(`six-${index}`, account, six[(index + 1) % six.length]!, index)
    );
    expect(patterns(sixEdges).filter((pattern) => pattern.pattern === 'cycle')).toHaveLength(0);
  });

  it('detects fan-in and fan-out from ten distinct counterparties in 72 hours', () => {
    const fanIn = Array.from({ length: 10 }, (_, index) =>
      transaction(`in-${index}`, `S${index}`, 'HUB', index)
    );
    const fanOut = Array.from({ length: 10 }, (_, index) =>
      transaction(`out-${index}`, 'HUB', `R${index}`, index)
    );
    expect(patterns(fanIn).some((pattern) => pattern.pattern === 'fan-in')).toBe(true);
    expect(patterns(fanOut).some((pattern) => pattern.pattern === 'fan-out')).toBe(true);
  });

  it('does not count repeated counterparties or transfers outside the 72-hour window as a fan pattern', () => {
    const repeated = [
      transaction('repeat-1', 'SAME', 'HUB', 0),
      transaction('repeat-2', 'SAME', 'HUB', 1),
      ...Array.from({ length: 9 }, (_, index) => transaction(`old-${index}`, `OLD${index}`, 'HUB', -100))
    ];
    expect(patterns(repeated).some((pattern) => pattern.pattern === 'fan-in')).toBe(false);
  });

  it('detects rapid pass-through and a layered chain only when intermediary flows are comparable and timely', () => {
    const chain = [
      transaction('ab', 'A', 'B', 0, 100),
      transaction('bc', 'B', 'C', 1, 95),
      transaction('cd', 'C', 'D', 2, 97)
    ];
    const findings = patterns(chain);
    expect(findings.some((pattern) => pattern.pattern === 'rapid-pass-through')).toBe(true);
    expect(findings.some((pattern) => pattern.pattern === 'layered-chain' && pattern.evidence.hopCount === 3)).toBe(true);

    const longDelay = [
      transaction('delay-in', 'A', 'B', 0, 100),
      transaction('delay-out', 'B', 'C', 30, 100)
    ];
    expect(patterns(longDelay).some((pattern) => pattern.pattern === 'rapid-pass-through')).toBe(false);
  });

  it('keeps disconnected graph components separate and preserves parallel transaction edges', () => {
    const graph = analyzeGraph([
      transaction('ab-1', 'A', 'B', 1),
      transaction('ab-2', 'A', 'B', 2),
      transaction('cd', 'C', 'D', 3)
    ]);
    expect(graph.graph.edges).toHaveLength(3);
    expect(graph.graph.edges.filter((edge) => edge.source === 'A' && edge.target === 'B')).toHaveLength(2);
    expect(graph.patterns.filter((pattern) => pattern.pattern === 'cycle')).toHaveLength(0);
  });

  it('counts self-transfers once while preserving both inbound and outbound graph degrees', () => {
    const graph = analyzeGraph([transaction('self', 'A', 'A', 1)]);
    expect(graph.graph.nodes).toHaveLength(1);
    expect(graph.graph.nodes[0]).toMatchObject({
      inDegree: 1,
      outDegree: 1,
      transactionCount: 1,
      incomingAmount: 100,
      outgoingAmount: 100
    });
  });

  it('reports high-velocity transfer evidence separately from AI assessment scores', () => {
    const transfers = Array.from({ length: 20 }, (_, index) =>
      transaction(`velocity-${index}`, 'SENDER', `R${index}`, index / 60, 10)
    );
    const result = analyzeGraph(transfers);
    expect(result.patterns.some((pattern) => pattern.pattern === 'high-velocity')).toBe(true);
    expect(result.graph.edges[0]?.riskAssessment.finalScore).toBe(20);
  });
});
