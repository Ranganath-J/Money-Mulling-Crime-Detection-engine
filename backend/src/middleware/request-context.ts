import { randomUUID } from 'node:crypto';
import type { NextFunction, Request, Response } from 'express';

export function requestContext(request: Request, response: Response, next: NextFunction): void {
  const requestId = request.header('x-request-id') ?? randomUUID();
  response.setHeader('x-request-id', requestId);
  const startedAt = Date.now();
  response.on('finish', () => {
    console.info(JSON.stringify({ requestId, method: request.method, path: request.originalUrl, status: response.statusCode, durationMs: Date.now() - startedAt }));
  });
  next();
}
