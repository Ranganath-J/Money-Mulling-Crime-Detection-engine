import { Pool } from 'pg';
import { env } from './env.js';

export const pool = env.DATABASE_URL
  ? new Pool({ connectionString: env.DATABASE_URL, connectionTimeoutMillis: env.DB_CONNECT_TIMEOUT_MS })
  : null;

export async function ensureSchema(): Promise<void> {
  if (!pool) throw new Error('DATABASE_URL is not configured');
  await pool.query(`
    CREATE TABLE IF NOT EXISTS transactions (
      id TEXT PRIMARY KEY,
      sender_account_id TEXT NOT NULL,
      receiver_account_id TEXT NOT NULL,
      amount NUMERIC(12, 2) NOT NULL CHECK (amount > 0 AND amount <= 10000000),
      currency VARCHAR(3) NOT NULL DEFAULT 'USD',
      occurred_at TIMESTAMPTZ NOT NULL,
      channel TEXT NOT NULL DEFAULT 'online',
      risk_assessment JSONB NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS transactions_occurred_at_idx ON transactions (occurred_at DESC);
    CREATE INDEX IF NOT EXISTS transactions_sender_occurred_at_idx ON transactions (sender_account_id, occurred_at DESC);
    CREATE INDEX IF NOT EXISTS transactions_receiver_occurred_at_idx ON transactions (receiver_account_id, occurred_at DESC);
    CREATE TABLE IF NOT EXISTS transaction_analyses (
      id UUID PRIMARY KEY,
      file_name TEXT NOT NULL,
      status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'succeeded', 'failed')),
      total_rows INTEGER NOT NULL DEFAULT 0,
      processed_rows INTEGER NOT NULL DEFAULT 0,
      accepted_rows INTEGER NOT NULL DEFAULT 0,
      rejected_rows INTEGER NOT NULL DEFAULT 0,
      result JSONB,
      error_message TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS transaction_analyses_created_at_idx ON transaction_analyses (created_at DESC);
    CREATE TABLE IF NOT EXISTS transaction_analysis_rows (
      analysis_id UUID NOT NULL REFERENCES transaction_analyses(id) ON DELETE CASCADE,
      row_number INTEGER NOT NULL,
      transaction_id TEXT,
      accepted BOOLEAN NOT NULL DEFAULT FALSE,
      raw_record JSONB,
      errors JSONB NOT NULL DEFAULT '[]'::jsonb,
      PRIMARY KEY (analysis_id, row_number)
    );
    CREATE INDEX IF NOT EXISTS transaction_analysis_rows_transaction_id_idx ON transaction_analysis_rows (transaction_id);
  `);
  await pool.query(`
    UPDATE transaction_analyses
    SET status = 'failed',
        error_message = 'Analysis was interrupted by a server restart. Upload the file again to retry.',
        updated_at = NOW()
    WHERE status IN ('queued', 'processing')
  `);
}

export async function checkDatabase(): Promise<{ ok: true } | { ok: false; reason: string }> {
  if (!pool) return { ok: false, reason: 'DATABASE_URL is not configured' };
  try {
    await pool.query('SELECT 1');
    return { ok: true };
  } catch (error) {
    const reason = error instanceof Error ? error.message : 'Unknown database error';
    return { ok: false, reason };
  }
}
