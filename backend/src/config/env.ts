import 'dotenv/config';
import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  PORT: z.coerce.number().int().min(1).max(65535).default(3000),
  DATABASE_URL: z.string().url().optional(),
  DB_CONNECT_TIMEOUT_MS: z.coerce.number().int().positive().max(30000).default(3000),
  AI_SERVICE_URL: z.string().url().default('http://127.0.0.1:8001'),
  CSV_MAX_FILE_BYTES: z.coerce.number().int().positive().max(50 * 1024 * 1024).default(10 * 1024 * 1024),
  CSV_MAX_ROWS: z.coerce.number().int().positive().max(50000).default(5000),
  GRAPH_FAN_WINDOW_HOURS: z.coerce.number().positive().max(8760).default(72),
  GRAPH_FAN_COUNTERPARTY_THRESHOLD: z.coerce.number().int().positive().max(10000).default(10),
  GRAPH_PASS_THROUGH_WINDOW_HOURS: z.coerce.number().positive().max(8760).default(24),
  GRAPH_PASS_THROUGH_AMOUNT_TOLERANCE: z.coerce.number().min(0).max(1).default(0.2),
  GRAPH_HIGH_VELOCITY_WINDOW_HOURS: z.coerce.number().positive().max(168).default(1),
  GRAPH_HIGH_VELOCITY_TRANSFER_THRESHOLD: z.coerce.number().int().positive().max(50000).default(20),
  GRAPH_HIGH_VELOCITY_AMOUNT_THRESHOLD: z.coerce.number().positive().max(1_000_000_000).default(100_000),
  LOG_LEVEL: z.string().default('info')
});

const parsed = envSchema.safeParse(process.env);
if (!parsed.success) {
  throw new Error(`Invalid environment configuration: ${parsed.error.message}`);
}

export const env = parsed.data;
