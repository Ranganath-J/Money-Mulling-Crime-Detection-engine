import { env } from '../config/env.js';
import type { TransactionInput } from '../schemas/transaction.js';

type Prediction = {
  status: string;
  score: number | null;
  model_version?: string;
  risk_category?: 'Monitor' | 'Review' | 'Investigation Alert' | null;
  reasons?: string[] | null;
  detected_topologies?: string[] | null;
  error?: string | null;
};

type HistoryRow = {
  id: string;
  sender_account_id: string;
  receiver_account_id: string;
  amount: string | number;
  occurred_at: Date | string;
  channel: string;
};

type GraphNode = {
  id: string;
  inflow_sum: number;
  outflow_sum: number;
  degree: number;
};

type GraphEdge = {
  transactionId: string;
  source: string;
  target: string;
  amount: number;
  timestamp: string;
  channel: string;
};

const componentWeights = {
  transaction_classifier: 0.4,
  anomaly_detector: 0.2,
  gnn_detector: 0.4
} as const;

function parseHistoryTimestamp(timestamp: Date | string): Date {
  return timestamp instanceof Date ? timestamp : new Date(timestamp);
}

function buildGraph(history: HistoryRow[], transaction: TransactionInput & { id: string; timestamp: string }): {
  nodes: GraphNode[];
  edges: GraphEdge[];
} {
  const edges: GraphEdge[] = [...history].reverse().map((row) => ({
    transactionId: row.id,
    source: row.sender_account_id,
    target: row.receiver_account_id,
    amount: Number(row.amount),
    timestamp: parseHistoryTimestamp(row.occurred_at).toISOString(),
    channel: row.channel
  }));
  edges.push({
    transactionId: transaction.id,
    source: transaction.senderAccountId,
    target: transaction.receiverAccountId,
    amount: transaction.amount,
    timestamp: transaction.timestamp,
    channel: transaction.channel
  });

  const nodes = new Map<string, GraphNode>();
  for (const edge of edges) {
    const source = nodes.get(edge.source) ?? { id: edge.source, inflow_sum: 0, outflow_sum: 0, degree: 0 };
    const target = nodes.get(edge.target) ?? { id: edge.target, inflow_sum: 0, outflow_sum: 0, degree: 0 };
    source.outflow_sum += edge.amount;
    source.degree += 1;
    target.inflow_sum += edge.amount;
    target.degree += 1;
    nodes.set(source.id, source);
    nodes.set(target.id, target);
  }
  return { nodes: [...nodes.values()], edges };
}

async function predict(path: string, body: unknown): Promise<Prediction> {
  const response = await fetch(`${env.AI_SERVICE_URL.replace(/\/+$/, '')}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(6000)
  });
  if (!response.ok) {
    throw new Error(`AI endpoint ${path} returned HTTP ${response.status}`);
  }
  return await response.json() as Prediction;
}

function usableScore(prediction: Prediction | null): number | null {
  if (
    !prediction
    || prediction.status === 'unavailable'
    || typeof prediction.score !== 'number'
    || !Number.isFinite(prediction.score)
    || prediction.score < 0
    || prediction.score > 100
  ) return null;
  return prediction.score;
}

function riskCategory(score: number): 'Monitor' | 'Review' | 'Investigation Alert' {
  if (score >= 75) return 'Investigation Alert';
  if (score >= 40) return 'Review';
  return 'Monitor';
}

export async function assessTransaction(
  transaction: TransactionInput & { id: string; timestamp: string },
  history: HistoryRow[],
  graphHistory = history
): Promise<{
  assessment: {
    status: 'available' | 'partial' | 'unavailable';
    finalScore: number | null;
    category: 'Monitor' | 'Review' | 'Investigation Alert' | null;
    componentScores: Record<string, number | null>;
    graphFindings: string[];
    reasons: string[];
    modelVersion: string | null;
  };
  aiErrors: string[];
}> {
  const outbound = history.filter((row) => row.sender_account_id === transaction.senderAccountId);
  const inbound = history.filter((row) => row.receiver_account_id === transaction.senderAccountId);
  const historicalAmounts = outbound.map((row) => Number(row.amount));
  const historicalAverage = historicalAmounts.length
    ? historicalAmounts.reduce((sum, amount) => sum + amount, 0) / historicalAmounts.length
    : 0;
  const historicalStd = historicalAmounts.length
    ? Math.sqrt(historicalAmounts.reduce((sum, amount) => sum + (amount - historicalAverage) ** 2, 0) / historicalAmounts.length)
    : 0;
  const timestampMs = new Date(transaction.timestamp).getTime();
  const velocity = outbound.filter((row) => {
    const age = timestampMs - parseHistoryTimestamp(row.occurred_at).getTime();
    return age >= 0 && age <= 60 * 60 * 1000;
  }).length;
  const inflow = inbound.reduce((sum, row) => sum + Number(row.amount), 0);
  const outflow = historicalAmounts.reduce((sum, amount) => sum + amount, 0);
  const earliest = history.reduce<number | null>((earliestMs, row) => {
    const time = parseHistoryTimestamp(row.occurred_at).getTime();
    return earliestMs === null || time < earliestMs ? time : earliestMs;
  }, null);
  const context = {
    recent_velocity: velocity,
    historical_avg_amount: historicalAverage,
    historical_std_amount: historicalStd,
    inflow_outflow_ratio: outflow > 0 ? inflow / outflow : 1,
    account_age_days: earliest === null ? 30 : Math.max(0, Math.floor((timestampMs - earliest) / 86_400_000))
  };
  const graph = buildGraph(graphHistory, transaction);
  const calls = [
    predict('/predict/transaction', {
      id: transaction.id,
      senderAccountId: transaction.senderAccountId,
      receiverAccountId: transaction.receiverAccountId,
      amount: transaction.amount,
      currency: transaction.currency,
      timestamp: transaction.timestamp,
      channel: transaction.channel,
      context
    }),
    predict('/predict/anomaly', {
      senderAccountId: transaction.senderAccountId,
      current_amount: transaction.amount,
      timestamp: transaction.timestamp,
      channel: transaction.channel,
      history: history.map((row) => ({
        amount: Number(row.amount),
        timestamp: parseHistoryTimestamp(row.occurred_at).toISOString(),
        direction: row.sender_account_id === transaction.senderAccountId ? 'outbound' : 'inbound',
        channel: row.channel
      }))
    }),
    predict('/predict/gnn', {
      targetTransactionId: transaction.id,
      nodes: graph.nodes,
      edges: graph.edges
    })
  ] as const;

  const results = await Promise.allSettled(calls);
  const predictions = results.map((result) => result.status === 'fulfilled' ? result.value : null);
  const aiErrors = results.flatMap((result, index) => {
    if (result.status === 'rejected') {
      const message = result.reason instanceof Error ? result.reason.message : 'Unknown AI service error';
      console.error(JSON.stringify({ component: Object.keys(componentWeights)[index], error: message }));
      return [message];
    }
    if (result.value.status === 'unavailable') {
      return [result.value.error ?? `Model ${Object.keys(componentWeights)[index]} is unavailable`];
    }
    return [];
  });

  const componentScores = {
    transaction_classifier: usableScore(predictions[0] ?? null),
    anomaly_detector: usableScore(predictions[1] ?? null),
    gnn_detector: usableScore(predictions[2] ?? null)
  };
  const activeWeight = Object.entries(componentWeights).reduce(
    (sum, [name, weight]) => sum + (componentScores[name as keyof typeof componentWeights] === null ? 0 : weight),
    0
  );
  const finalScore = activeWeight > 0
    ? Math.round(Object.entries(componentWeights).reduce(
      (sum, [name, weight]) => sum + (componentScores[name as keyof typeof componentWeights] ?? 0) * weight,
      0
    ) / activeWeight * 10) / 10
    : null;
  const reasons = [...new Set(predictions.flatMap((prediction) => prediction?.reasons ?? []))];
  const graphFindings = [...new Set(predictions[2]?.detected_topologies ?? [])];
  if (aiErrors.length && finalScore === null) reasons.push('No AI model returned an available risk score.');
  if (aiErrors.length && finalScore !== null) reasons.push('Some AI models were unavailable; score uses available model components only.');

  return {
    assessment: {
      status: finalScore === null ? 'unavailable' : aiErrors.length ? 'partial' : 'available',
      finalScore,
      category: finalScore === null ? null : riskCategory(finalScore),
      componentScores,
      graphFindings,
      reasons: [...new Set(reasons)],
      modelVersion: predictions.find((prediction) => prediction?.model_version)?.model_version ?? null
    },
    aiErrors
  };
}

export async function getAiHealth(): Promise<{ status: string; models?: Record<string, unknown> } | null> {
  try {
    const response = await fetch(`${env.AI_SERVICE_URL.replace(/\/+$/, '')}/health`, {
      signal: AbortSignal.timeout(2000)
    });
    if (!response.ok) throw new Error(`AI health endpoint returned HTTP ${response.status}`);
    return await response.json() as { status: string; models?: Record<string, unknown> };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown AI health error';
    console.error(JSON.stringify({ service: 'ai-health', error: message }));
    return null;
  }
}
