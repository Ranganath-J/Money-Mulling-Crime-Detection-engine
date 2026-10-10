import { Pool } from 'pg';
import { env } from './env.js';

export const pool = env.DATABASE_URL
  ? new Pool({ connectionString: env.DATABASE_URL, connectionTimeoutMillis: env.DB_CONNECT_TIMEOUT_MS })
  : null;

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
