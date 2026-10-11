import { Router } from 'express';
import { checkDatabase } from '../config/database.js';

export const healthRouter = Router();

healthRouter.get('/health', (_request, response) => {
  response.json({ status: 'ok', service: 'money-muling-detection-backend' });
});

healthRouter.get('/health/db', async (_request, response) => {
  const result = await checkDatabase();
  if (!result.ok) {
    response.status(503).json({ status: 'unavailable', database: result.reason });
    return;
  }
  response.json({ status: 'ok', database: 'connected' });
});
