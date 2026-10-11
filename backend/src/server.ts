import { createServer } from 'node:http';
import { createApp } from './app.js';
import { env } from './config/env.js';
import { ensureSchema, pool } from './config/database.js';

const server = createServer(createApp());
void ensureSchema().then(() => {
  server.listen(env.PORT, () => console.info(`Backend listening on port ${env.PORT}`));
}).catch(async (error: unknown) => {
  console.error('Backend startup failed: PostgreSQL is required and the transaction schema could not be initialized.', error);
  await pool?.end();
  process.exitCode = 1;
});

async function shutdown(signal: string): Promise<void> {
  console.info(`Received ${signal}; shutting down`);
  server.close(async () => {
    await pool?.end();
    process.exit(0);
  });
}

process.once('SIGINT', () => void shutdown('SIGINT'));
process.once('SIGTERM', () => void shutdown('SIGTERM'));
