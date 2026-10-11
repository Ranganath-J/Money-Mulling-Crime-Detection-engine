import request from 'supertest';
import { describe, expect, it } from 'vitest';
import { createApp } from '../src/app.js';
import { transactionInputSchema } from '../src/schemas/transaction.js';

const app = createApp();

describe('transaction API validation', () => {
  it('accepts standard currency amounts with two decimal places', () => {
    const result = transactionInputSchema.safeParse({
      senderAccountId: 'ACC-A',
      receiverAccountId: 'ACC-B',
      amount: 12.34,
      currency: 'usd'
    });

    expect(result.success).toBe(true);
    if (result.success) expect(result.data.currency).toBe('USD');
  });

  it('rejects incomplete or out-of-range transaction payloads', async () => {
    const response = await request(app)
      .post('/api/transactions')
      .send({ senderAccountId: 'ACC-A', receiverAccountId: 'ACC-B', amount: 10_000_001 });

    expect(response.status).toBe(400);
    expect(response.body.error.code).toBe('INVALID_TRANSACTION');
  });

  it('rejects invalid transaction history limits', async () => {
    const response = await request(app).get('/api/transactions?limit=201');

    expect(response.status).toBe(400);
    expect(response.body.error.code).toBe('INVALID_LIMIT');
  });

  it('returns a client error for malformed JSON', async () => {
    const response = await request(app)
      .post('/api/transactions')
      .set('Content-Type', 'application/json')
      .send('{');

    expect(response.status).toBe(400);
    expect(response.body.error.code).toBe('INVALID_JSON');
  });
});
