import { randomUUID } from 'node:crypto';
import { Router } from 'express';
import { checkDatabase, pool } from '../config/database.js';
import { assessTransaction, getAiHealth } from '../services/ai-client.js';
import { rowToTransaction, transactionInputSchema } from '../schemas/transaction.js';

export const transactionsRouter = Router();

transactionsRouter.get('/transactions', async (request, response, next) => {
  const limitValue = request.query.limit === undefined ? 100 : Number(request.query.limit);
  if (!Number.isInteger(limitValue) || limitValue < 1 || limitValue > 200) {
    response.status(400).json({ error: { code: 'INVALID_LIMIT', message: 'limit must be an integer between 1 and 200' } });
    return;
  }
  if (!pool) {
    response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
    return;
  }
  try {
    const result = await pool.query(
      'SELECT * FROM transactions ORDER BY occurred_at DESC, created_at DESC LIMIT $1',
      [limitValue]
    );
    response.json({ transactions: result.rows.map(rowToTransaction) });
  } catch (error) {
    next(error);
  }
});

transactionsRouter.post('/transactions', async (request, response, next) => {
  const parsed = transactionInputSchema.safeParse(request.body);
  if (!parsed.success) {
    response.status(400).json({
      error: {
        code: 'INVALID_TRANSACTION',
        message: parsed.error.issues.map((issue) => `${issue.path.join('.') || 'body'}: ${issue.message}`).join('; ')
      }
    });
    return;
  }
  if (!pool) {
    response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
    return;
  }

  const transaction = {
    ...parsed.data,
    id: parsed.data.id ?? randomUUID(),
    timestamp: parsed.data.timestamp ?? new Date().toISOString()
  };
  try {
    const [historyResult, graphHistoryResult] = await Promise.all([
      pool.query(
        `SELECT id, sender_account_id, receiver_account_id, amount, occurred_at, channel
         FROM transactions
         WHERE (sender_account_id = $1 OR receiver_account_id = $1)
           AND occurred_at < $2
         ORDER BY occurred_at DESC
         LIMIT 1000`,
        [transaction.senderAccountId, transaction.timestamp]
      ),
      pool.query(
        `SELECT id, sender_account_id, receiver_account_id, amount, occurred_at, channel
         FROM transactions
         WHERE (sender_account_id = ANY($1::text[]) OR receiver_account_id = ANY($1::text[]))
           AND occurred_at < $2
         ORDER BY occurred_at DESC
         LIMIT 1000`,
        [[transaction.senderAccountId, transaction.receiverAccountId], transaction.timestamp]
      )
    ]);
    const { assessment, aiErrors } = await assessTransaction(transaction, historyResult.rows, graphHistoryResult.rows);
    const result = await pool.query(
      `INSERT INTO transactions (
         id, sender_account_id, receiver_account_id, amount, currency, occurred_at, channel, risk_assessment
       ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
       RETURNING *`,
      [
        transaction.id,
        transaction.senderAccountId,
        transaction.receiverAccountId,
        transaction.amount,
        transaction.currency,
        transaction.timestamp,
        transaction.channel,
        JSON.stringify(assessment)
      ]
    );
    response.status(201).json({
      transaction: rowToTransaction(result.rows[0] as Record<string, unknown>),
      warnings: aiErrors
    });
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === '23505') {
      response.status(409).json({ error: { code: 'DUPLICATE_TRANSACTION', message: `Transaction ${transaction.id} already exists` } });
      return;
    }
    next(error);
  }
});

transactionsRouter.get('/dashboard', async (_request, response, next) => {
  if (!pool) {
    response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
    return;
  }
  try {
    const [counts, recent] = await Promise.all([
      pool.query(`
        SELECT
          COUNT(*)::int AS total,
          COUNT(*) FILTER (WHERE risk_assessment->>'category' = 'Monitor')::int AS monitor,
          COUNT(*) FILTER (WHERE risk_assessment->>'category' = 'Review')::int AS review,
          COUNT(*) FILTER (WHERE risk_assessment->>'category' = 'Investigation Alert')::int AS alert
        FROM transactions
      `),
      pool.query('SELECT * FROM transactions ORDER BY occurred_at DESC, created_at DESC LIMIT 10')
    ]);
    const row = counts.rows[0] as { total: number; monitor: number; review: number; alert: number };
    response.json({
      totalTransactions: row.total,
      monitorCount: row.monitor,
      reviewCount: row.review,
      alertCount: row.alert,
      recentTransactions: recent.rows.map(rowToTransaction)
    });
  } catch (error) {
    next(error);
  }
});

transactionsRouter.get('/status', async (_request, response, next) => {
  if (!pool) {
    response.status(503).json({
      backend: 'ok',
      database: 'unavailable',
      aiService: 'unknown'
    });
    return;
  }
  try {
    const database = await checkDatabase();
    const aiHealth = await getAiHealth();
    response.json({
      backend: 'ok',
      database: database.ok ? 'ok' : 'unavailable',
      aiService: aiHealth?.status ?? 'unavailable',
      models: aiHealth?.models ?? {}
    });
  } catch (error) {
    next(error);
  }
});
