import { randomUUID } from 'node:crypto';
import { Router, type ErrorRequestHandler, type NextFunction, type Request, type RequestHandler, type Response } from 'express';
import multer from 'multer';
import { pool } from '../config/database.js';
import {
  CsvInputError,
  CSV_LIMITS,
  parseCsv,
  type CsvValidationError
} from '../services/csv-analysis.js';
import { processAnalysisJob } from '../services/csv-analysis-jobs.js';

export const analysisRouter = Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: CSV_LIMITS.maxFileBytes, files: 1, fields: 5 }
});

const uploadErrorHandler: ErrorRequestHandler = (error, _request, response, next) => {
  if (error instanceof CsvInputError) {
    response.status(error.statusCode).json({ error: { code: error.code, message: error.message } });
    return;
  }
  if (error instanceof multer.MulterError) {
    const fileTooLarge = error.code === 'LIMIT_FILE_SIZE';
    response.status(fileTooLarge ? 413 : 400).json({
      error: {
        code: fileTooLarge ? 'FILE_TOO_LARGE' : 'INVALID_UPLOAD',
        message: fileTooLarge ? 'CSV files must be 10 MB or smaller.' : 'Upload one CSV file in the "file" field.'
      }
    });
    return;
  }
  next(error);
};

const singleCsvUpload: RequestHandler = (request, response, next) => {
  upload.single('file')(request, response, (error: unknown) => {
    if (error) next(error);
    else next();
  });
};

function getValidationErrors(value: unknown): CsvValidationError[] {
  if (!Array.isArray(value)) return [];
  return value as CsvValidationError[];
}

function isAnalysisResult(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

analysisRouter.post('/analysis/upload', singleCsvUpload, uploadErrorHandler, async (
  request: Request,
  response: Response,
  next: NextFunction
) => {
  if (!pool) {
    response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
    return;
  }
  if (!request.file) {
    response.status(400).json({ error: { code: 'MISSING_FILE', message: 'Choose a CSV file in the "file" field.' } });
    return;
  }
  if (!request.file.originalname.toLowerCase().endsWith('.csv')) {
    response.status(415).json({ error: { code: 'UNSUPPORTED_FILE_TYPE', message: 'Upload a .csv file.' } });
    return;
  }

  try {
    const parsed = parseCsv(request.file.buffer);
    const analysisId = randomUUID();
    const fileName = request.file.originalname.split(/[\\/]/).at(-1)?.slice(0, 200) || 'transactions.csv';
    const stagedRows = [
      ...parsed.transactions.map((transaction) => ({
        row_number: transaction.rowNumber,
        transaction_id: transaction.id,
        raw_record: transaction,
        errors: []
      })),
      ...parsed.validationErrors.map((issue) => ({
        row_number: issue.rowNumber,
        transaction_id: issue.transactionId ?? null,
        raw_record: null,
        errors: issue.errors
      }))
    ].sort((a, b) => a.row_number - b.row_number);

    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      await client.query(
        `INSERT INTO transaction_analyses (id, file_name, status, total_rows, rejected_rows)
         VALUES ($1, $2, 'queued', $3, $4)`,
        [analysisId, fileName, parsed.totalRows, parsed.validationErrors.length]
      );
      if (stagedRows.length) {
        await client.query(
          `INSERT INTO transaction_analysis_rows (analysis_id, row_number, transaction_id, raw_record, errors)
           SELECT $1, row.row_number, row.transaction_id, row.raw_record, row.errors
           FROM jsonb_to_recordset($2::jsonb) AS row(
             row_number INTEGER, transaction_id TEXT, raw_record JSONB, errors JSONB
           )`,
          [analysisId, JSON.stringify(stagedRows)]
        );
      }
      await client.query('COMMIT');
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }

    setImmediate(() => {
      void processAnalysisJob(
        analysisId,
        fileName,
        parsed.totalRows,
        parsed.transactions,
        parsed.validationErrors
      ).catch((error: unknown) => {
        console.error(JSON.stringify({
          analysisId,
          service: 'csv-analysis-job',
          error: error instanceof Error ? error.message : 'Unknown job failure'
        }));
      });
    });

    response.status(202).json({
      analysis: {
        id: analysisId,
        fileName,
        status: 'queued',
        totalRows: parsed.totalRows,
        processedRows: parsed.validationErrors.length,
        acceptedRows: 0,
        rejectedRows: parsed.validationErrors.length,
        validationErrors: parsed.validationErrors
      }
    });
  } catch (error) {
    if (error instanceof CsvInputError) {
      response.status(error.statusCode).json({ error: { code: error.code, message: error.message } });
      return;
    }
    next(error);
  }
});

analysisRouter.get('/analysis', async (request, response, next) => {
  const limitValue = request.query.limit === undefined ? 20 : Number(request.query.limit);
  if (!Number.isInteger(limitValue) || limitValue < 1 || limitValue > 50) {
    response.status(400).json({ error: { code: 'INVALID_LIMIT', message: 'limit must be an integer between 1 and 50' } });
    return;
  }
  if (!pool) {
    response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
    return;
  }
  try {
    const result = await pool.query(
      `SELECT id, file_name, status, total_rows, processed_rows, accepted_rows, rejected_rows,
              result->'summary' AS summary, error_message, created_at, updated_at
       FROM transaction_analyses
       ORDER BY created_at DESC
       LIMIT $1`,
      [limitValue]
    );
    response.json({
      analyses: result.rows.map((row) => ({
        id: row.id,
        fileName: row.file_name,
        status: row.status,
        totalRows: row.total_rows,
        processedRows: row.processed_rows,
        acceptedRows: row.accepted_rows,
        rejectedRows: row.rejected_rows,
        summary: row.summary,
        error: row.error_message,
        createdAt: row.created_at,
        updatedAt: row.updated_at
      }))
    });
  } catch (error) {
    next(error);
  }
});

analysisRouter.get('/analysis/:analysisId', async (request, response, next) => {
  if (!pool) {
    response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
    return;
  }
  try {
    const result = await pool.query(
      `SELECT id, file_name, status, total_rows, processed_rows, accepted_rows, rejected_rows,
              result, error_message, created_at, updated_at
       FROM transaction_analyses
       WHERE id = $1`,
      [request.params.analysisId]
    );
    const row = result.rows[0] as Record<string, unknown> | undefined;
    if (!row) {
      response.status(404).json({ error: { code: 'ANALYSIS_NOT_FOUND', message: 'Analysis was not found.' } });
      return;
    }
    const analysisResult = isAnalysisResult(row.result) ? row.result : null;
    response.json({
      analysis: {
        id: row.id,
        fileName: row.file_name,
        status: row.status,
        totalRows: row.total_rows,
        processedRows: row.processed_rows,
        acceptedRows: row.accepted_rows,
        rejectedRows: row.rejected_rows,
        summary: analysisResult?.summary ?? null,
        thresholds: analysisResult?.thresholds ?? null,
        processingStats: analysisResult?.processingStats ?? null,
        validationErrors: getValidationErrors(analysisResult?.validationErrors),
        interpretation: analysisResult?.interpretation ?? [],
        error: row.error_message,
        createdAt: row.created_at,
        updatedAt: row.updated_at
      }
    });
  } catch (error) {
    next(error);
  }
});

analysisRouter.get('/analysis/:analysisId/graph', async (request, response, next) => {
  const limitValue = request.query.limit === undefined ? 1000 : Number(request.query.limit);
  const offsetValue = request.query.offset === undefined ? 0 : Number(request.query.offset);
  if (!Number.isInteger(limitValue) || limitValue < 1 || limitValue > 5000 || !Number.isInteger(offsetValue) || offsetValue < 0) {
    response.status(400).json({ error: { code: 'INVALID_PAGINATION', message: 'limit must be 1–5000 and offset must be zero or greater.' } });
    return;
  }
  if (!pool) {
    response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
    return;
  }
  try {
    const result = await pool.query('SELECT status, result FROM transaction_analyses WHERE id = $1', [request.params.analysisId]);
    const row = result.rows[0] as { status: string; result: unknown } | undefined;
    if (!row) {
      response.status(404).json({ error: { code: 'ANALYSIS_NOT_FOUND', message: 'Analysis was not found.' } });
      return;
    }
    if (row.status !== 'succeeded' || !isAnalysisResult(row.result)) {
      response.status(409).json({ error: { code: 'ANALYSIS_NOT_READY', message: 'Graph results are available after analysis completes.' } });
      return;
    }
    const graph = row.result.graph as { nodes?: unknown[]; edges?: unknown[] } | undefined;
    const edges = graph?.edges ?? [];
    response.json({
      nodes: graph?.nodes ?? [],
      edges: edges.slice(offsetValue, offsetValue + limitValue),
      totalNodes: graph?.nodes?.length ?? 0,
      totalEdges: edges.length,
      limit: limitValue,
      offset: offsetValue
    });
  } catch (error) {
    next(error);
  }
});

for (const resource of ['patterns', 'accounts'] as const) {
  analysisRouter.get(`/analysis/:analysisId/${resource}`, async (request, response, next) => {
    if (!pool) {
      response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
      return;
    }
    try {
      const query = resource === 'patterns'
        ? 'SELECT status, result->\'patterns\' AS value FROM transaction_analyses WHERE id = $1'
        : 'SELECT status, result->\'accounts\' AS value FROM transaction_analyses WHERE id = $1';
      const result = await pool.query(query, [request.params.analysisId]);
      const row = result.rows[0] as { status: string; value: unknown } | undefined;
      if (!row) {
        response.status(404).json({ error: { code: 'ANALYSIS_NOT_FOUND', message: 'Analysis was not found.' } });
        return;
      }
      if (row.status !== 'succeeded') {
        response.status(409).json({ error: { code: 'ANALYSIS_NOT_READY', message: 'Results are available after analysis completes.' } });
        return;
      }
      response.json(resource === 'patterns' ? { patterns: row.value ?? [] } : { accounts: row.value ?? [] });
    } catch (error) {
      next(error);
    }
  });
}

analysisRouter.get('/analysis/:analysisId/export', async (request, response, next) => {
  if (!pool) {
    response.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'DATABASE_URL is not configured' } });
    return;
  }
  try {
    const result = await pool.query(
      'SELECT status, result FROM transaction_analyses WHERE id = $1',
      [request.params.analysisId]
    );
    const row = result.rows[0] as { status: string; result: unknown } | undefined;
    if (!row) {
      response.status(404).json({ error: { code: 'ANALYSIS_NOT_FOUND', message: 'Analysis was not found.' } });
      return;
    }
    if (row.status !== 'succeeded' || !isAnalysisResult(row.result)) {
      response.status(409).json({ error: { code: 'ANALYSIS_NOT_READY', message: 'Export is available after analysis completes.' } });
      return;
    }
    response
      .type('application/json')
      .set('Content-Disposition', `attachment; filename="analysis-${request.params.analysisId}.json"`)
      .send(`${JSON.stringify(row.result, null, 2)}\n`);
  } catch (error) {
    next(error);
  }
});
