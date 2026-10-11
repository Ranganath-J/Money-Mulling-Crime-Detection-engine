import { createHash } from 'node:crypto';
import { parse } from 'csv-parse/sync';
import { env } from '../config/env.js';
import type { RiskAssessment } from '../schemas/transaction.js';

export const CSV_LIMITS = {
  maxFileBytes: env.CSV_MAX_FILE_BYTES,
  maxRows: env.CSV_MAX_ROWS,
  maxIdentifierLength: 120,
  maxAmount: 10_000_000
} as const;

export const PATTERN_THRESHOLDS = {
  cycleMinAccounts: 3,
  cycleMaxAccounts: 5,
  cycleMaxFindings: 200,
  cycleSearchSteps: 1_000_000,
  fanWindowHours: env.GRAPH_FAN_WINDOW_HOURS,
  fanCounterpartyThreshold: env.GRAPH_FAN_COUNTERPARTY_THRESHOLD,
  passThroughWindowHours: env.GRAPH_PASS_THROUGH_WINDOW_HOURS,
  passThroughAmountTolerance: env.GRAPH_PASS_THROUGH_AMOUNT_TOLERANCE,
  passThroughComparisons: 1_000_000,
  layeredChainMinHops: 3,
  layeredChainMaxHops: 5,
  highVelocityWindowHours: env.GRAPH_HIGH_VELOCITY_WINDOW_HOURS,
  highVelocityTransactionThreshold: env.GRAPH_HIGH_VELOCITY_TRANSFER_THRESHOLD,
  highVelocityAmountThreshold: env.GRAPH_HIGH_VELOCITY_AMOUNT_THRESHOLD,
  maxFindings: 500
} as const;

const requiredHeaders = ['transaction_id', 'sender_id', 'receiver_id', 'amount', 'timestamp'] as const;

export type CsvValidationError = {
  rowNumber: number;
  transactionId?: string;
  errors: string[];
};

export type CsvTransaction = {
  rowNumber: number;
  id: string;
  senderAccountId: string;
  receiverAccountId: string;
  amount: number;
  timestamp: string;
  storageTimestamp: string;
  sourceTimestamp: string;
};

export class CsvInputError extends Error {
  constructor(
    message: string,
    readonly statusCode: number,
    readonly code: string
  ) {
    super(message);
    this.name = 'CsvInputError';
  }
}

function parseTimestamp(value: string): { normalized: string; storage: string } | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z|[+-]\d{2}:\d{2})?$/i.exec(value);
  if (!match) return null;
  const [, yearText, monthText, dayText, hourText, minuteText, secondText, fraction = '', zone = ''] = match;
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const hour = Number(hourText);
  const minute = Number(minuteText);
  const second = Number(secondText);
  const calendar = new Date(Date.UTC(year, month - 1, day, hour, minute, second));
  if (
    calendar.getUTCFullYear() !== year
    || calendar.getUTCMonth() !== month - 1
    || calendar.getUTCDate() !== day
    || hour > 23
    || minute > 59
    || second > 59
  ) return null;
  if (zone && zone.toUpperCase() !== 'Z') {
    const offsetParts = zone.slice(1).split(':');
    if (offsetParts.length !== 2) return null;
    const offsetHours = Number(offsetParts[0]);
    const offsetMinutes = Number(offsetParts[1]);
    if (offsetHours > 14 || offsetMinutes > 59 || (offsetHours === 14 && offsetMinutes !== 0)) return null;
  }
  const storage = `${yearText}-${monthText}-${dayText}T${hourText}:${minuteText}:${secondText}${fraction ? `.${fraction}` : ''}${zone || 'Z'}`;
  const normalized = `${yearText}-${monthText}-${dayText}T${hourText}:${minuteText}:${secondText}${fraction ? `.${fraction.padEnd(3, '0').slice(0, 3)}` : ''}${zone || 'Z'}`;
  const timestamp = new Date(normalized);
  return Number.isFinite(timestamp.getTime()) ? { normalized: timestamp.toISOString(), storage } : null;
}

function validIdentifier(value: string): boolean {
  return value.length > 0
    && value.length <= CSV_LIMITS.maxIdentifierLength
    && !/[\u0000-\u001f\u007f]/.test(value);
}

export function parseCsv(buffer: Buffer): {
  transactions: CsvTransaction[];
  validationErrors: CsvValidationError[];
  totalRows: number;
} {
  if (buffer.length === 0) throw new CsvInputError('The uploaded CSV file is empty.', 400, 'EMPTY_FILE');
  if (buffer.length > CSV_LIMITS.maxFileBytes) {
    throw new CsvInputError('CSV files must be 10 MB or smaller.', 413, 'FILE_TOO_LARGE');
  }

  let text: string;
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(buffer);
  } catch {
    throw new CsvInputError('The CSV must use UTF-8 encoding.', 415, 'UNSUPPORTED_ENCODING');
  }

  let records: string[][];
  try {
    records = parse(text, {
      bom: true,
      skip_empty_lines: false,
      relax_column_count: true,
      record_delimiter: ['\r\n', '\n', '\r']
    }) as string[][];
  } catch {
    throw new CsvInputError('The CSV structure is malformed. Check quoting and delimiters.', 400, 'MALFORMED_CSV');
  }
  if (records.length === 0 || records[0]?.every((value) => !value.trim())) {
    throw new CsvInputError('The CSV has no header row.', 400, 'MISSING_HEADERS');
  }
  if (records.length - 1 > CSV_LIMITS.maxRows) {
    throw new CsvInputError(`A CSV may contain at most ${CSV_LIMITS.maxRows} data rows.`, 413, 'TOO_MANY_ROWS');
  }

  const headerValues = (records[0] ?? []).map((value) => value.trim().toLowerCase());
  const duplicateHeaders = headerValues.filter((value, index) => value && headerValues.indexOf(value) !== index);
  if (duplicateHeaders.length) {
    throw new CsvInputError('The CSV contains duplicate column headers.', 400, 'DUPLICATE_HEADERS');
  }
  const missingHeaders = requiredHeaders.filter((header) => !headerValues.includes(header));
  if (missingHeaders.length) {
    throw new CsvInputError(`Missing required CSV columns: ${missingHeaders.join(', ')}.`, 400, 'MISSING_HEADERS');
  }
  const indices = Object.fromEntries(requiredHeaders.map((header) => [header, headerValues.indexOf(header)])) as Record<typeof requiredHeaders[number], number>;
  const seenIds = new Set<string>();
  const transactions: CsvTransaction[] = [];
  const validationErrors: CsvValidationError[] = [];

  records.slice(1).forEach((record, index) => {
    const rowNumber = index + 2;
    const values = requiredHeaders.map((header) => (record[indices[header]] ?? '').trim());
    const id = values[0] ?? '';
    const senderAccountId = values[1] ?? '';
    const receiverAccountId = values[2] ?? '';
    const amountText = values[3] ?? '';
    const sourceTimestamp = values[4] ?? '';
    const errors: string[] = [];
    if (!record.some((value) => value.trim())) {
      errors.push('Row is empty.');
    } else if (record.length !== headerValues.length) {
      errors.push(`Expected ${headerValues.length} columns but found ${record.length}.`);
    }
    if (!id || !validIdentifier(id)) errors.push('Transaction ID must be 1–120 characters and contain no control characters.');
    if (!senderAccountId || !validIdentifier(senderAccountId)) errors.push('Sender ID must be 1–120 characters and contain no control characters.');
    if (!receiverAccountId || !validIdentifier(receiverAccountId)) errors.push('Receiver ID must be 1–120 characters and contain no control characters.');
    if (!/^\d+(?:\.\d{1,2})?$/.test(amountText)) {
      errors.push('Amount must be a positive number with no more than two decimal places.');
    } else {
      const amount = Number(amountText);
      if (!Number.isFinite(amount) || amount <= 0 || amount > CSV_LIMITS.maxAmount) {
        errors.push(`Amount must be greater than 0 and no more than ${CSV_LIMITS.maxAmount}.`);
      }
    }
    const timestamp = parseTimestamp(sourceTimestamp);
    if (!timestamp) errors.push('Timestamp must be a valid date and time (ISO format or YYYY-MM-DD HH:mm:ss).');
    if (id && seenIds.has(id)) errors.push(`Duplicate transaction ID "${id}" in this file.`);
    if (id) seenIds.add(id);

    if (errors.length) {
      validationErrors.push({ rowNumber, ...(id ? { transactionId: id } : {}), errors });
      return;
    }
    transactions.push({
      rowNumber,
      id,
      senderAccountId,
      receiverAccountId,
      amount: Number(amountText),
      timestamp: timestamp!.normalized,
      storageTimestamp: timestamp!.storage,
      sourceTimestamp
    });
  });

  return { transactions, validationErrors, totalRows: records.length - 1 };
}

export type ScoredCsvTransaction = CsvTransaction & {
  assessment: RiskAssessment;
};

export type AnalysisPattern = {
  id: string;
  pattern: 'cycle' | 'fan-in' | 'fan-out' | 'layered-chain' | 'rapid-pass-through' | 'high-velocity';
  severity: 'medium' | 'high';
  description: string;
  accountIds: string[];
  edgeIds: string[];
  evidence: Record<string, unknown>;
};

export type AnalysisGraphEdge = {
  id: string;
  source: string;
  target: string;
  transactionId: string;
  amount: number;
  timestamp: string;
  sourceTimestamp: string;
  riskAssessment: RiskAssessment;
  patternIds: string[];
};

export type AnalysisGraphNode = {
  id: string;
  inDegree: number;
  outDegree: number;
  transactionCount: number;
  incomingAmount: number;
  outgoingAmount: number;
  patternCount: number;
  riskScore: number | null;
  componentScores: Record<string, number | null>;
};

export type SuspiciousAccount = AnalysisGraphNode & {
  patternIds: string[];
};

function findingId(pattern: AnalysisPattern['pattern'], edgeIds: string[], accountIds: string[]): string {
  const identity = `${pattern}|${accountIds.join('>')}|${[...edgeIds].sort().join('|')}`;
  return `${pattern}-${createHash('sha256').update(identity).digest('hex').slice(0, 16)}`;
}

export function analyzeGraph(
  transactions: ScoredCsvTransaction[],
  options: { maxFindings?: number } = {}
): {
  graph: { nodes: AnalysisGraphNode[]; edges: AnalysisGraphEdge[] };
  patterns: AnalysisPattern[];
  accounts: SuspiciousAccount[];
  warnings: string[];
} {
  const maxFindings = options.maxFindings ?? PATTERN_THRESHOLDS.maxFindings;
  const warnings: string[] = [];
  const patterns: AnalysisPattern[] = [];
  const edges = [...transactions].sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id));
  const edgeById = new Map(edges.map((edge) => [edge.id, edge]));
  const adjacency = new Map<string, ScoredCsvTransaction[]>();
  const incoming = new Map<string, ScoredCsvTransaction[]>();
  const outgoing = new Map<string, ScoredCsvTransaction[]>();
  const addPattern = (
    pattern: AnalysisPattern['pattern'],
    severity: AnalysisPattern['severity'],
    description: string,
    accountIds: string[],
    edgeIds: string[],
    evidence: Record<string, unknown>
  ): boolean => {
    if (patterns.length >= maxFindings) return false;
    patterns.push({
      id: findingId(pattern, edgeIds, accountIds),
      pattern,
      severity,
      description,
      accountIds,
      edgeIds,
      evidence
    });
    return true;
  };
  for (const edge of edges) {
    const from = outgoing.get(edge.senderAccountId) ?? [];
    from.push(edge);
    outgoing.set(edge.senderAccountId, from);
    const to = incoming.get(edge.receiverAccountId) ?? [];
    to.push(edge);
    incoming.set(edge.receiverAccountId, to);
    const links = adjacency.get(edge.senderAccountId) ?? [];
    links.push(edge);
    adjacency.set(edge.senderAccountId, links);
  }
  for (const links of adjacency.values()) links.sort((a, b) => a.receiverAccountId.localeCompare(b.receiverAccountId) || a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id));

  const accountIds = [...new Set(edges.flatMap((edge) => [edge.senderAccountId, edge.receiverAccountId]))].sort();
  let cycleSteps = 0;
  let cycleSearchTruncated = false;
  const cycleKeys = new Set<string>();
  for (const start of accountIds) {
    const pathAccounts = [start];
    const pathEdges: ScoredCsvTransaction[] = [];
    const walk = (current: string): void => {
      if (patterns.length >= Math.min(maxFindings, PATTERN_THRESHOLDS.cycleMaxFindings)) return;
      const targets = new Set<string>();
      for (const edge of adjacency.get(current) ?? []) {
        if (cycleSteps++ >= PATTERN_THRESHOLDS.cycleSearchSteps) {
          cycleSearchTruncated = true;
          return;
        }
        const next = edge.receiverAccountId;
        if (next === start && pathAccounts.length >= PATTERN_THRESHOLDS.cycleMinAccounts) {
          const key = pathAccounts.join('>');
          if (!cycleKeys.has(key)) {
            cycleKeys.add(key);
            addPattern(
              'cycle',
              'high',
              `Circular routing was found across ${pathAccounts.length} distinct accounts.`,
              [...pathAccounts],
              [...pathEdges.map((item) => item.id), edge.id],
              {
                cycleLength: pathAccounts.length,
                orderedAccounts: [...pathAccounts, start],
                transactions: [...pathEdges, edge].map((item) => ({
                  transactionId: item.id,
                  amount: item.amount,
                  timestamp: item.timestamp
                }))
              }
            );
          }
          continue;
        }
        if (
          targets.has(next)
          || next.localeCompare(start) < 0
          || pathAccounts.includes(next)
          || pathAccounts.length >= PATTERN_THRESHOLDS.cycleMaxAccounts
        ) continue;
        targets.add(next);
        pathAccounts.push(next);
        pathEdges.push(edge);
        walk(next);
        pathEdges.pop();
        pathAccounts.pop();
        if (cycleSearchTruncated || patterns.length >= Math.min(maxFindings, PATTERN_THRESHOLDS.cycleMaxFindings)) return;
      }
    };
    walk(start);
    if (cycleSearchTruncated || patterns.length >= Math.min(maxFindings, PATTERN_THRESHOLDS.cycleMaxFindings)) break;
  }
  if (cycleSearchTruncated) warnings.push('Cycle search reached its configured step limit; some cycles may not be reported.');
  else if (patterns.filter((pattern) => pattern.pattern === 'cycle').length >= PATTERN_THRESHOLDS.cycleMaxFindings) {
    warnings.push('Cycle reporting reached its configured finding limit; additional cycles may exist.');
  }

  const fanWindowMs = PATTERN_THRESHOLDS.fanWindowHours * 3_600_000;
  const inspectFan = (direction: 'in' | 'out', groups: Map<string, ScoredCsvTransaction[]>): void => {
    for (const [accountId, items] of groups) {
      const sorted = [...items].sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id));
      let startIndex = 0;
      let bestStart = 0;
      let bestEnd = -1;
      let bestCounterparties: string[] = [];
      const counterparties = new Map<string, number>();
      for (let endIndex = 0; endIndex < sorted.length; endIndex++) {
        const endEdge = sorted[endIndex]!;
        const endTime = Date.parse(endEdge.timestamp);
        const endCounterparty =
          direction === 'in' ? endEdge.senderAccountId : endEdge.receiverAccountId;
        counterparties.set(endCounterparty, (counterparties.get(endCounterparty) ?? 0) + 1);
        while (startIndex < endIndex && endTime - Date.parse(sorted[startIndex]!.timestamp) > fanWindowMs) {
          const oldEdge = sorted[startIndex]!;
          const oldCounterparty = direction === 'in' ? oldEdge.senderAccountId : oldEdge.receiverAccountId;
          const remaining = (counterparties.get(oldCounterparty) ?? 1) - 1;
          if (remaining === 0) counterparties.delete(oldCounterparty);
          else counterparties.set(oldCounterparty, remaining);
          startIndex++;
        }
        if (counterparties.size > bestCounterparties.length) {
          bestStart = startIndex;
          bestEnd = endIndex;
          bestCounterparties = [...counterparties.keys()];
        }
      }
      if (bestCounterparties.length < PATTERN_THRESHOLDS.fanCounterpartyThreshold) continue;
      const bestEdges = sorted.slice(bestStart, bestEnd + 1);
      const pattern = direction === 'in' ? 'fan-in' : 'fan-out';
      addPattern(
        pattern,
        'high',
        direction === 'in'
          ? `This account received funds from ${bestCounterparties.length} distinct senders within ${PATTERN_THRESHOLDS.fanWindowHours} hours.`
          : `This account sent funds to ${bestCounterparties.length} distinct receivers within ${PATTERN_THRESHOLDS.fanWindowHours} hours.`,
        [accountId, ...[...bestCounterparties].sort()],
        bestEdges.map((edge) => edge.id),
        {
          distinctCounterparties: bestCounterparties.length,
          transferCount: bestEdges.length,
          windowHours: PATTERN_THRESHOLDS.fanWindowHours,
          startsAt: bestEdges[0]?.timestamp,
          endsAt: bestEdges.at(-1)?.timestamp
        }
      );
    }
  };
  inspectFan('in', incoming);
  inspectFan('out', outgoing);

  const passWindowMs = PATTERN_THRESHOLDS.passThroughWindowHours * 3_600_000;
  const amountTolerance = PATTERN_THRESHOLDS.passThroughAmountTolerance;
  const passPairs = new Map<string, ScoredCsvTransaction[]>();
  let passComparisons = 0;
  let passSearchTruncated = false;
  for (const [accountId, received] of incoming) {
    const receivedSorted = [...received].sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id));
    const sent = [...(outgoing.get(accountId) ?? [])].sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id));
    let firstPossible = 0;
    let accountPassCount = 0;
    let firstPassPair: { inboundEdge: ScoredCsvTransaction; outboundEdge: ScoredCsvTransaction } | undefined;
    for (const inboundEdge of receivedSorted) {
      const inboundTime = Date.parse(inboundEdge.timestamp);
      while (firstPossible < sent.length && Date.parse(sent[firstPossible]!.timestamp) < inboundTime) firstPossible++;
      for (let index = firstPossible; index < sent.length; index++) {
        const outboundEdge = sent[index]!;
        const age = Date.parse(outboundEdge.timestamp) - inboundTime;
        if (age > passWindowMs) break;
        if (passComparisons++ >= PATTERN_THRESHOLDS.passThroughComparisons) {
          passSearchTruncated = true;
          break;
        }
        const ratio = outboundEdge.amount / inboundEdge.amount;
        if (Math.abs(ratio - 1) > amountTolerance) continue;
        const matches = passPairs.get(inboundEdge.id) ?? [];
        matches.push(outboundEdge);
        passPairs.set(inboundEdge.id, matches);
        accountPassCount++;
        firstPassPair ??= { inboundEdge, outboundEdge };
        if (patterns.length >= maxFindings) break;
      }
      if (passSearchTruncated || patterns.length >= maxFindings) break;
    }
    if (accountPassCount > 0 && patterns.length < maxFindings) {
      const { inboundEdge, outboundEdge } = firstPassPair!;
      addPattern(
        'rapid-pass-through',
        'medium',
        `Funds were forwarded by ${accountId} within ${PATTERN_THRESHOLDS.passThroughWindowHours} hours at a comparable amount.`,
        [inboundEdge.senderAccountId, accountId, outboundEdge.receiverAccountId],
        [inboundEdge.id, outboundEdge.id],
        {
          accountId,
          receivedAmount: inboundEdge.amount,
          forwardedAmount: outboundEdge.amount,
          amountDifferencePercent: Math.round(Math.abs(outboundEdge.amount / inboundEdge.amount - 1) * 1000) / 10,
          elapsedMinutes: Math.round((Date.parse(outboundEdge.timestamp) - Date.parse(inboundEdge.timestamp)) / 60_000),
          windowHours: PATTERN_THRESHOLDS.passThroughWindowHours,
          amountTolerancePercent: amountTolerance * 100
        }
      );
    }
  }
  if (passSearchTruncated) warnings.push('Pass-through matching reached its configured comparison limit; some matches may not be reported.');

  let chainSteps = 0;
  let chainSearchTruncated = false;
  const chainKeys = new Set<string>();
  const followChain = (first: ScoredCsvTransaction, chain: ScoredCsvTransaction[], visited: Set<string>): void => {
    if (chain.length >= PATTERN_THRESHOLDS.layeredChainMinHops) {
      const nodes = [chain[0]!.senderAccountId, ...chain.map((edge) => edge.receiverAccountId)];
      const key = nodes.join('>');
      if (!chainKeys.has(key)) {
        chainKeys.add(key);
        addPattern(
          'layered-chain',
          'high',
          `A ${chain.length}-hop route contains intermediary accounts with comparable, time-near pass-through transfers.`,
          nodes,
          chain.map((edge) => edge.id),
          {
            hopCount: chain.length,
            intermediaryAccounts: nodes.slice(1, -1),
            passThroughWindowHours: PATTERN_THRESHOLDS.passThroughWindowHours,
            amountTolerancePercent: amountTolerance * 100
          }
        );
      }
    }
    if (chain.length >= PATTERN_THRESHOLDS.layeredChainMaxHops || patterns.length >= maxFindings) return;
    const last = chain.at(-1) ?? first;
    for (const next of passPairs.get(last.id) ?? []) {
      if (chainSteps++ >= PATTERN_THRESHOLDS.cycleSearchSteps) {
        chainSearchTruncated = true;
        return;
      }
      if (visited.has(next.receiverAccountId)) continue;
      visited.add(next.receiverAccountId);
      chain.push(next);
      followChain(first, chain, visited);
      chain.pop();
      visited.delete(next.receiverAccountId);
      if (chainSearchTruncated || patterns.length >= maxFindings) return;
    }
  };
  for (const [firstId, matches] of passPairs) {
    const first = edgeById.get(firstId);
    if (!first) continue;
    for (const second of matches) {
      if (patterns.length >= maxFindings || chainSearchTruncated) break;
      followChain(first, [first, second], new Set([first.senderAccountId, first.receiverAccountId, second.receiverAccountId]));
    }
  }
  if (chainSearchTruncated) warnings.push('Layered-chain search reached its configured step limit; some routes may not be reported.');

  const velocityWindowMs = PATTERN_THRESHOLDS.highVelocityWindowHours * 3_600_000;
  for (const [accountId, items] of outgoing) {
    const sorted = [...items].sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id));
    let left = 0;
    let bestStart = 0;
    let bestEnd = -1;
    let bestAmount = 0;
    let bestSignal = 0;
    let currentAmount = 0;
    for (let right = 0; right < sorted.length; right++) {
      currentAmount += sorted[right]!.amount;
      while (left < right && Date.parse(sorted[right]!.timestamp) - Date.parse(sorted[left]!.timestamp) > velocityWindowMs) {
        currentAmount -= sorted[left]!.amount;
        left++;
      }
      const signal = Math.max(
        (right - left + 1) / PATTERN_THRESHOLDS.highVelocityTransactionThreshold,
        currentAmount / PATTERN_THRESHOLDS.highVelocityAmountThreshold
      );
      if (signal > bestSignal) {
        bestStart = left;
        bestEnd = right;
        bestAmount = currentAmount;
        bestSignal = signal;
      }
    }
    if (bestEnd < bestStart) continue;
    const selected = sorted.slice(bestStart, bestEnd + 1);
    if (
      selected.length < PATTERN_THRESHOLDS.highVelocityTransactionThreshold
      && bestAmount < PATTERN_THRESHOLDS.highVelocityAmountThreshold
    ) continue;
    addPattern(
      'high-velocity',
      'medium',
      `${accountId} sent ${selected.length} transfers totaling ${bestAmount.toFixed(2)} within ${PATTERN_THRESHOLDS.highVelocityWindowHours} hour(s).`,
      [accountId, ...[...new Set(selected.map((edge) => edge.receiverAccountId))].sort()],
      selected.map((edge) => edge.id),
      {
        transferCount: selected.length,
        aggregateAmount: Math.round(bestAmount * 100) / 100,
        windowHours: PATTERN_THRESHOLDS.highVelocityWindowHours,
        thresholds: {
          transactionCount: PATTERN_THRESHOLDS.highVelocityTransactionThreshold,
          aggregateAmount: PATTERN_THRESHOLDS.highVelocityAmountThreshold
        }
      }
    );
  }
  if (patterns.length >= maxFindings) warnings.push('Pattern reporting reached its configured finding limit; additional findings may exist.');

  const patternsByAccount = new Map<string, string[]>();
  const patternsByEdge = new Map<string, string[]>();
  for (const pattern of patterns) {
    for (const accountId of pattern.accountIds) {
      const values = patternsByAccount.get(accountId) ?? [];
      values.push(pattern.id);
      patternsByAccount.set(accountId, values);
    }
    for (const edgeId of pattern.edgeIds) {
      const values = patternsByEdge.get(edgeId) ?? [];
      values.push(pattern.id);
      patternsByEdge.set(edgeId, values);
    }
  }

  const nodeStats = new Map<string, AnalysisGraphNode>();
  const riskTotals = new Map<string, { total: number; count: number }>();
  const getNode = (id: string): AnalysisGraphNode => {
    const existing = nodeStats.get(id);
    if (existing) return existing;
    const node: AnalysisGraphNode = {
      id,
      inDegree: 0,
      outDegree: 0,
      transactionCount: 0,
      incomingAmount: 0,
      outgoingAmount: 0,
      patternCount: 0,
      riskScore: null,
      componentScores: {}
    };
    nodeStats.set(id, node);
    return node;
  };
  for (const edge of edges) {
    const sender = getNode(edge.senderAccountId);
    const receiver = getNode(edge.receiverAccountId);
    sender.outDegree++;
    sender.transactionCount++;
    sender.outgoingAmount += edge.amount;
    receiver.inDegree++;
    if (sender !== receiver) receiver.transactionCount++;
    receiver.incomingAmount += edge.amount;
    for (const node of sender === receiver ? [sender] : [sender, receiver]) {
      const score = edge.assessment.finalScore;
      if (score !== null) {
        const aggregate = riskTotals.get(node.id) ?? { total: 0, count: 0 };
        aggregate.total += score;
        aggregate.count++;
        riskTotals.set(node.id, aggregate);
      }
      for (const [component, componentScore] of Object.entries(edge.assessment.componentScores)) {
        if (componentScore === null) continue;
        node.componentScores[component] = Math.max(node.componentScores[component] ?? 0, componentScore);
      }
      nodeStats.set(node.id, node);
    }
  }
  const nodes = [...nodeStats.values()].map((node) => ({
    ...node,
    incomingAmount: Math.round(node.incomingAmount * 100) / 100,
    outgoingAmount: Math.round(node.outgoingAmount * 100) / 100,
    riskScore: riskTotals.has(node.id)
      ? Math.round((riskTotals.get(node.id)!.total / riskTotals.get(node.id)!.count) * 10) / 10
      : null,
    patternCount: patternsByAccount.get(node.id)?.length ?? 0
  })).sort((a, b) => a.id.localeCompare(b.id));
  const graphEdges: AnalysisGraphEdge[] = edges.map((edge) => ({
    id: edge.id,
    source: edge.senderAccountId,
    target: edge.receiverAccountId,
    transactionId: edge.id,
    amount: edge.amount,
    timestamp: edge.timestamp,
    sourceTimestamp: edge.sourceTimestamp,
    riskAssessment: edge.assessment,
    patternIds: patternsByEdge.get(edge.id) ?? []
  }));
  const accounts = nodes
    .filter((node) => node.patternCount > 0 || (node.riskScore !== null && node.riskScore >= 40))
    .map((node) => ({ ...node, patternIds: patternsByAccount.get(node.id) ?? [] }))
    .sort((a, b) => (b.riskScore ?? -1) - (a.riskScore ?? -1) || b.patternCount - a.patternCount || a.id.localeCompare(b.id));

  return {
    graph: { nodes, edges: graphEdges },
    patterns,
    accounts,
    warnings
  };
}
