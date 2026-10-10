import type { ErrorRequestHandler, RequestHandler } from 'express';

export const notFound: RequestHandler = (request, response) => {
  response.status(404).json({ error: { code: 'NOT_FOUND', message: `Route ${request.method} ${request.path} was not found` } });
};

export const errorHandler: ErrorRequestHandler = (error, request, response, _next) => {
  console.error(JSON.stringify({ requestId: response.getHeader('x-request-id'), error: error instanceof Error ? error.message : 'Unknown error' }));
  response.status(500).json({ error: { code: 'INTERNAL_ERROR', message: 'An unexpected error occurred' } });
};
