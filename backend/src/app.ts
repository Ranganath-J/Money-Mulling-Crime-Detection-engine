import express from 'express';
import { errorHandler, notFound } from './middleware/error-handler.js';
import { requestContext } from './middleware/request-context.js';
import { healthRouter } from './routes/health.routes.js';

export function createApp() {
  const app = express();
  app.disable('x-powered-by');
  app.use(express.json({ limit: '1mb' }));
  app.use(requestContext);
  app.use(healthRouter);
  app.use(notFound);
  app.use(errorHandler);
  return app;
}
