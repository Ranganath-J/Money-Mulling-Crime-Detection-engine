import request from 'supertest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createApp } from '../src/app.js';

const mocks = vi.hoisted(() => ({
  databaseQuery: vi.fn(),
  clientQuery: vi.fn(),
  processAnalysisJob: vi.fn()
}));

vi.mock('../src/config/database.js', () => ({
  pool: {
    query: mocks.databaseQuery,
    connect: vi.fn(async () => ({
      query: mocks.clientQuery,
      release: vi.fn()
    }))
  },
  checkDatabase: vi.fn(async () => ({ ok: true }))
}));

vi.mock('../src/services/csv-analysis-jobs.js', () => ({
  processAnalysisJob: mocks.processAnalysisJob
}));

const app = createApp();
const validCsv = [
  'transaction_id,sender_id,receiver_id,amount,timestamp',
  'UPLOAD_TX_1,ACC_A,ACC_B,5000,2026-02-18 10:00:00'
].join('\n');

beforeEach(() => {
  mocks.databaseQuery.mockReset().mockResolvedValue({ rows: [] });
  mocks.clientQuery.mockReset().mockResolvedValue({ rows: [] });
  mocks.processAnalysisJob.mockReset().mockResolvedValue(undefined);
});

afterEach(() => vi.restoreAllMocks());

describe('CSV analysis upload and export routes', () => {
  it('accepts a valid multipart CSV upload and returns an analysis job ID', async () => {
    const response = await request(app)
      .post('/api/analysis/upload')
      .attach('file', Buffer.from(validCsv), { filename: 'transfers.csv', contentType: 'text/csv' });

    expect(response.status).toBe(202);
    expect(response.body.analysis).toMatchObject({
      fileName: 'transfers.csv',
      status: 'queued',
      totalRows: 1,
      validationErrors: []
    });
    expect(response.body.analysis.id).toMatch(/^[0-9a-f-]{36}$/i);
    expect(mocks.clientQuery).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO transaction_analyses'),
      expect.any(Array)
    );
    await vi.waitFor(() => expect(mocks.processAnalysisJob).toHaveBeenCalledOnce());
  });

  it('returns a row-level validation error and does not omit rejected input rows', async () => {
    const response = await request(app)
      .post('/api/analysis/upload')
      .attach('file', Buffer.from([
        'timestamp,sender_id,transaction_id,amount,receiver_id',
        '2026-02-18 10:00:00,ACC_A,UPLOAD_BAD,0,ACC_B'
      ].join('\n')), { filename: 'bad-rows.csv', contentType: 'text/csv' });

    expect(response.status).toBe(202);
    expect(response.body.analysis).toMatchObject({
      totalRows: 1,
      rejectedRows: 1,
      validationErrors: [{
        rowNumber: 2,
        transactionId: 'UPLOAD_BAD'
      }]
    });
  });

  it('rejects malformed and missing-column CSVs without creating analysis jobs', async () => {
    const malformed = await request(app)
      .post('/api/analysis/upload')
      .attach('file', Buffer.from('transaction_id,sender_id\n"broken'), { filename: 'bad.csv' });
    expect(malformed.status).toBe(400);
    expect(malformed.body.error.code).toBe('MALFORMED_CSV');

    const missingHeader = await request(app)
      .post('/api/analysis/upload')
      .attach('file', Buffer.from('transaction_id,sender_id\nTX-1,ACC_A'), { filename: 'missing.csv' });
    expect(missingHeader.status).toBe(400);
    expect(missingHeader.body.error.code).toBe('MISSING_HEADERS');
  });

  it('exports the versioned report as valid JSON', async () => {
    const report = {
      schemaVersion: '1.0.0',
      analysisId: 'test-analysis',
      status: 'succeeded',
      graph: { nodes: [], edges: [] },
      patterns: [],
      accounts: [],
      validationErrors: []
    };
    mocks.databaseQuery.mockResolvedValueOnce({ rows: [{ status: 'succeeded', result: report }] });

    const response = await request(app).get('/api/analysis/test-analysis/export');
    expect(response.status).toBe(200);
    expect(response.headers['content-type']).toContain('application/json');
    expect(response.headers['content-disposition']).toContain('analysis-test-analysis.json');
    expect(JSON.parse(response.text)).toEqual(report);
  });
});
