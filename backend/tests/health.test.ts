import request from 'supertest';
import { describe, expect, it } from 'vitest';
import { createApp } from '../src/app.js';

const app = createApp();

describe('health and error handling', () => {
  it('returns stable process health', async () => {
    const response = await request(app).get('/health');
    expect(response.status).toBe(200);
    expect(response.body).toEqual({ status: 'ok', service: 'money-muling-detection-backend' });
    expect(response.headers['x-request-id']).toBeTruthy();
  });

  it('returns a structured not-found error', async () => {
    const response = await request(app).get('/does-not-exist');
    expect(response.status).toBe(404);
    expect(response.body.error.code).toBe('NOT_FOUND');
  });

  it('reports unavailable database configuration honestly', async () => {
    const response = await request(app).get('/health/db');
    expect([200, 503]).toContain(response.status);
    if (response.status === 503) expect(response.body.status).toBe('unavailable');
  });
});
