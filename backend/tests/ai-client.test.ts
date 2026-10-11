import { afterEach, describe, expect, it, vi } from 'vitest';
import { assessTransaction } from '../src/services/ai-client.js';

const transaction = {
  id: 'tx-test',
  senderAccountId: 'account-a',
  receiverAccountId: 'account-b',
  amount: 125,
  currency: 'USD',
  channel: 'online',
  timestamp: '2026-10-11T10:00:00.000Z'
};

function prediction(status: string, score: number | null, extra: Record<string, unknown> = {}) {
  return new Response(JSON.stringify({
    status,
    score,
    model_version: '1.0.0',
    reasons: [],
    ...extra
  }), { status: 200, headers: { 'content-type': 'application/json' } });
}

afterEach(() => vi.restoreAllMocks());

describe('AI scoring integration', () => {
  it('combines all available model scores using the documented weights', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/predict/transaction')) return prediction('available', 50);
      if (url.endsWith('/predict/anomaly')) return prediction('available', 50);
      return prediction('available', 100, { detected_topologies: ['cycle'] });
    });

    const result = await assessTransaction(transaction, []);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(result.assessment).toMatchObject({
      status: 'available',
      finalScore: 70,
      category: 'Review',
      componentScores: {
        transaction_classifier: 50,
        anomaly_detector: 50,
        gnn_detector: 100
      }
    });
    expect(result.assessment.graphFindings).toContain('cycle');
    expect(result.aiErrors).toEqual([]);
  });

  it('keeps the final score null when every model is unavailable', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      prediction('unavailable', null, { error: 'model not loaded' }));

    const result = await assessTransaction(transaction, []);

    expect(result.assessment.status).toBe('unavailable');
    expect(result.assessment.finalScore).toBeNull();
    expect(result.assessment.category).toBeNull();
    expect(Object.values(result.assessment.componentScores)).toEqual([null, null, null]);
    expect(result.aiErrors).toHaveLength(3);
  });

  it('reweights only the model components that returned a score', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/predict/transaction')) return prediction('available', 40);
      if (url.endsWith('/predict/anomaly')) return prediction('unavailable', null);
      return prediction('available', 80);
    });

    const result = await assessTransaction(transaction, []);

    expect(result.assessment.status).toBe('partial');
    expect(result.assessment.finalScore).toBe(60);
    expect(result.assessment.componentScores.anomaly_detector).toBeNull();
  });
});
