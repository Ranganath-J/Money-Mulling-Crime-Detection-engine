import { pool } from '../config/database.js';
import type { RiskAssessment } from '../schemas/transaction.js';
import { assessTransaction } from './ai-client.js';
import {
  analyzeGraph,
  PATTERN_THRESHOLDS,
  type CsvTransaction,
  type CsvValidationError,
  type ScoredCsvTransaction
} from './csv-analysis.js';

type HistoryRow = {
  id: string;
  sender_account_id: string;
  receiver_account_id: string;
  amount: string | number;
  occurred_at: Date | string;
  channel: string;
};

type ExistingTransactionRow = HistoryRow & {
  risk_assessment: unknown;
  matches_upload: boolean;
};

type ScoredRow = {
  transaction: CsvTransaction;
  assessment: RiskAssessment;
};

type AnalysisSummary = {
  totalRows: number;
  acceptedRows: number;
  rejectedRows: number;
  nodeCount: number;
  edgeCount: number;
  patternCounts: Record<string, number>;
  scoredTransactionCount: number;
  unscoredTransactionCount: number;
};

function mapLimit<T, R>(items: T[], limit: number, worker: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let nextIndex = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex++;
      results[index] = await worker(items[index]!);
    }
  });
  return Promise.all(runners).then(() => results);
}

function rowsBefore(rows: HistoryRow[], timestamp: string, limit: number): HistoryRow[] {
  const target = Date.parse(timestamp);
  let low = 0;
  let high = rows.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    const value = rows[middle]!.occurred_at instanceof Date
      ? rows[middle]!.occurred_at.getTime()
      : Date.parse(rows[middle]!.occurred_at);
    if (value < target) low = middle + 1;
    else high = middle;
  }
  return rows.slice(Math.max(0, low - limit), low);
}

function toAssessment(value: unknown): RiskAssessment {
  return value as RiskAssessment;
}

function makeSummary(
  totalRows: number,
  acceptedRows: number,
  graph: ReturnType<typeof analyzeGraph>['graph'],
  patterns: ReturnType<typeof analyzeGraph>['patterns']
): AnalysisSummary {
  const patternCounts: Record<string, number> = {
    cycle: 0,
    'fan-in': 0,
    'fan-out': 0,
    'layered-chain': 0,
    'rapid-pass-through': 0,
    'high-velocity': 0
  };
  for (const pattern of patterns) patternCounts[pattern.pattern] = (patternCounts[pattern.pattern] ?? 0) + 1;
  const scoredTransactionCount = graph.edges.filter((edge) => edge.riskAssessment.finalScore !== null).length;
  return {
    totalRows,
    acceptedRows,
    rejectedRows: totalRows - acceptedRows,
    nodeCount: graph.nodes.length,
    edgeCount: graph.edges.length,
    patternCounts,
    scoredTransactionCount,
    unscoredTransactionCount: graph.edges.length - scoredTransactionCount
  };
}

async function updateProgress(analysisId: string, processedRows: number): Promise<void> {
  if (!pool) throw new Error('Database pool is unavailable');
  await pool.query(
    'UPDATE transaction_analyses SET status = $2, processed_rows = $3, updated_at = NOW() WHERE id = $1',
    [analysisId, processedRows > 0 ? 'processing' : 'queued', processedRows]
  );
}

export async function processAnalysisJob(
  analysisId: string,
  fileName: string,
  totalRows: number,
  candidates: CsvTransaction[],
  initialErrors: CsvValidationError[]
): Promise<void> {
  if (!pool) throw new Error('Database pool is unavailable');
  try {
    await pool.query(
      "UPDATE transaction_analyses SET status = 'processing', processed_rows = $2, updated_at = NOW() WHERE id = $1",
      [analysisId, initialErrors.length]
    );

    const uniqueIds = [...new Set(candidates.map((item) => item.id))];
    const existingResult = uniqueIds.length
      ? await pool.query<ExistingTransactionRow>(
        `SELECT stored.id, stored.sender_account_id, stored.receiver_account_id, stored.amount,
                stored.occurred_at, stored.channel, stored.risk_assessment,
                stored.sender_account_id = incoming.sender_account_id
                  AND stored.receiver_account_id = incoming.receiver_account_id
                  AND stored.amount = incoming.amount
                  AND stored.occurred_at = incoming.occurred_at AS matches_upload
         FROM transactions AS stored
         JOIN jsonb_to_recordset($1::jsonb) AS incoming(
           id TEXT, sender_account_id TEXT, receiver_account_id TEXT, amount NUMERIC, occurred_at TIMESTAMPTZ
         ) ON incoming.id = stored.id`,
        [JSON.stringify(candidates.map((item) => ({
          id: item.id,
          sender_account_id: item.senderAccountId,
          receiver_account_id: item.receiverAccountId,
          amount: item.amount,
          occurred_at: item.storageTimestamp
        })))]
      )
      : { rows: [] };
    const existingById = new Map(existingResult.rows.map((row) => [row.id, row]));
    const existingIds = new Set(existingById.keys());
    const alreadyStored = candidates.filter((item) => existingIds.has(item.id));
    const eligible = candidates.filter((item) => !existingIds.has(item.id));
    if (alreadyStored.length) {
      const duplicateRows = alreadyStored.map((item) => ({
        row_number: item.rowNumber,
        message: existingById.get(item.id)!.matches_upload
          ? `Transaction ID "${item.id}" already exists in persisted transactions. It was not re-imported; its saved assessment is included in this graph analysis.`
          : `Transaction ID "${item.id}" already exists with different transaction details. This row was not imported or included in this graph analysis.`
      }));
      await pool.query(
        `UPDATE transaction_analysis_rows AS rows
         SET errors = rows.errors || jsonb_build_array(duplicates.message)
         FROM jsonb_to_recordset($2::jsonb) AS duplicates(row_number INTEGER, message TEXT)
         WHERE rows.analysis_id = $1 AND rows.row_number = duplicates.row_number`,
        [analysisId, JSON.stringify(duplicateRows)]
      );
    }

    const accountIds = [...new Set(eligible.flatMap((item) => [item.senderAccountId, item.receiverAccountId]))];
    const liveHistoryResult = accountIds.length
      ? await pool.query<HistoryRow>(
        `SELECT id, sender_account_id, receiver_account_id, amount, occurred_at, channel
         FROM transactions
         WHERE sender_account_id = ANY($1::text[]) OR receiver_account_id = ANY($1::text[])
         ORDER BY occurred_at DESC
         LIMIT 10000`,
        [accountIds]
      )
      : { rows: [] };
    const rowsByAccount = new Map<string, HistoryRow[]>();
    const addHistory = (row: HistoryRow): void => {
      for (const id of new Set([row.sender_account_id, row.receiver_account_id])) {
        const rows = rowsByAccount.get(id) ?? [];
        rows.push(row);
        rowsByAccount.set(id, rows);
      }
    };
    for (const row of liveHistoryResult.rows) addHistory(row);
    for (const item of eligible) {
      addHistory({
        id: item.id,
        sender_account_id: item.senderAccountId,
        receiver_account_id: item.receiverAccountId,
        amount: item.amount,
        occurred_at: item.timestamp,
        channel: 'csv_import'
      });
    }
    for (const rows of rowsByAccount.values()) {
      rows.sort((a, b) => {
        const timeA = a.occurred_at instanceof Date ? a.occurred_at.getTime() : Date.parse(a.occurred_at);
        const timeB = b.occurred_at instanceof Date ? b.occurred_at.getTime() : Date.parse(b.occurred_at);
        return timeA - timeB || a.id.localeCompare(b.id);
      });
    }

    const sortedEligible = [...eligible].sort((a, b) => a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id));
    const scoredRows: ScoredRow[] = [];
    let aiErrorCount = 0;
    const scoreBatchSize = 20;
    for (let offset = 0; offset < sortedEligible.length; offset += scoreBatchSize) {
      const batch = sortedEligible.slice(offset, offset + scoreBatchSize);
      const scoredBatch = await mapLimit(batch, 4, async (item): Promise<ScoredRow> => {
        const senderHistory = rowsBefore(rowsByAccount.get(item.senderAccountId) ?? [], item.timestamp, 1000)
          .filter((row) => row.id !== item.id);
        const graphRows = [...new Map([
          ...(rowsByAccount.get(item.senderAccountId) ?? []),
          ...(rowsByAccount.get(item.receiverAccountId) ?? [])
        ].map((row) => [row.id, row])).values()]
          .filter((row) => {
            const value = row.occurred_at instanceof Date ? row.occurred_at.getTime() : Date.parse(row.occurred_at);
            return value < Date.parse(item.timestamp) && row.id !== item.id;
          })
          .sort((a, b) => {
            const timeA = a.occurred_at instanceof Date ? a.occurred_at.getTime() : Date.parse(a.occurred_at);
            const timeB = b.occurred_at instanceof Date ? b.occurred_at.getTime() : Date.parse(b.occurred_at);
            return timeB - timeA;
          })
          .slice(0, 1000);
        const { assessment, aiErrors } = await assessTransaction(
          {
            id: item.id,
            senderAccountId: item.senderAccountId,
            receiverAccountId: item.receiverAccountId,
            amount: item.amount,
            currency: 'USD',
            channel: 'csv_import',
            timestamp: item.timestamp
          },
          senderHistory,
          graphRows
        );
        aiErrorCount += aiErrors.length;
        return { transaction: item, assessment: toAssessment(assessment) };
      });
      scoredRows.push(...scoredBatch);
      await updateProgress(analysisId, Math.min(totalRows, initialErrors.length + alreadyStored.length + scoredRows.length));
    }

    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      const transactionRows = scoredRows.map(({ transaction, assessment }) => ({
        id: transaction.id,
        sender_account_id: transaction.senderAccountId,
        receiver_account_id: transaction.receiverAccountId,
        amount: transaction.amount,
        currency: 'USD',
        occurred_at: transaction.storageTimestamp,
        channel: 'csv_import',
        risk_assessment: assessment
      }));
      const insertResult = transactionRows.length
        ? await client.query<{ id: string }>(
          `INSERT INTO transactions (
             id, sender_account_id, receiver_account_id, amount, currency, occurred_at, channel, risk_assessment
           )
           SELECT item.id, item.sender_account_id, item.receiver_account_id, item.amount,
                  item.currency, item.occurred_at, item.channel, item.risk_assessment
           FROM jsonb_to_recordset($1::jsonb) AS item(
             id TEXT, sender_account_id TEXT, receiver_account_id TEXT, amount NUMERIC,
             currency TEXT, occurred_at TIMESTAMPTZ, channel TEXT, risk_assessment JSONB
           )
           ON CONFLICT (id) DO NOTHING
           RETURNING id`,
          [JSON.stringify(transactionRows)]
        )
        : { rows: [] };
      const insertedIds = new Set(insertResult.rows.map((row) => row.id));
      const insertedRows = scoredRows.filter(({ transaction }) => insertedIds.has(transaction.id));
      const racedDuplicates = scoredRows.filter(({ transaction }) => !insertedIds.has(transaction.id));
      if (insertedRows.length) {
        await client.query(
          `UPDATE transaction_analysis_rows AS rows
           SET accepted = TRUE
           FROM jsonb_to_recordset($2::jsonb) AS accepted(row_number INTEGER, transaction_id TEXT)
           WHERE rows.analysis_id = $1
             AND rows.row_number = accepted.row_number
             AND rows.transaction_id = accepted.transaction_id`,
          [analysisId, JSON.stringify(insertedRows.map(({ transaction }) => ({
            row_number: transaction.rowNumber,
            transaction_id: transaction.id
          })))]
        );
      }
      if (racedDuplicates.length) {
        await client.query(
          `UPDATE transaction_analysis_rows AS rows
           SET errors = rows.errors || jsonb_build_array('Transaction ID already exists in persisted transactions.')
           FROM jsonb_to_recordset($2::jsonb) AS duplicate(row_number INTEGER, transaction_id TEXT)
           WHERE rows.analysis_id = $1
             AND rows.row_number = duplicate.row_number
             AND rows.transaction_id = duplicate.transaction_id`,
          [analysisId, JSON.stringify(racedDuplicates.map(({ transaction }) => ({
            row_number: transaction.rowNumber,
            transaction_id: transaction.id
          })))]
        );
      }

      const analyzedTransactions: ScoredCsvTransaction[] = insertedRows.map(({ transaction, assessment }) => ({
        ...transaction,
        assessment
      }));
      for (const transaction of alreadyStored) {
        const existing = existingById.get(transaction.id)!;
        if (existing.matches_upload) {
          analyzedTransactions.push({
            ...transaction,
            assessment: toAssessment(existing.risk_assessment)
          });
        }
      }
      const result = analyzeGraph(analyzedTransactions);
      const summary = makeSummary(totalRows, insertedRows.length, result.graph, result.patterns);
      const validationResult = await client.query<{ row_number: number; transaction_id: string | null; errors: string[] }>(
        `SELECT row_number, transaction_id, errors
         FROM transaction_analysis_rows
         WHERE analysis_id = $1 AND jsonb_array_length(errors) > 0
         ORDER BY row_number`,
        [analysisId]
      );
      const validationErrors = validationResult.rows.map((row) => ({
        rowNumber: row.row_number,
        ...(row.transaction_id ? { transactionId: row.transaction_id } : {}),
        errors: row.errors
      }));
      const output = {
        schemaVersion: '1.0.0',
        analysisId,
        status: 'succeeded',
        fileName,
        createdAt: new Date().toISOString(),
        summary,
        thresholds: PATTERN_THRESHOLDS,
        processingStats: {
          rowsProcessed: totalRows,
          aiModelErrors: aiErrorCount,
          warnings: result.warnings
        },
        graph: result.graph,
        patterns: result.patterns,
        accounts: result.accounts,
        validationErrors,
        interpretation: [
          'Pattern findings are deterministic indicators for review, not proof of wrongdoing.',
          'AI component scores use the established transaction 40%, behavioral anomaly 20%, and GNN 40% weights when all components are available.',
          'Unavailable component scores remain null and available components are reweighted; deterministic rule findings do not change the fused AI score.',
          'Transaction timestamps without an explicit UTC offset were interpreted as UTC. The original timestamp text is retained on each graph edge.'
        ]
      };
      await client.query(
        `UPDATE transaction_analyses
         SET status = 'succeeded',
             processed_rows = $2,
             accepted_rows = $3,
             rejected_rows = $4,
             result = $5::jsonb,
             error_message = NULL,
             updated_at = NOW()
         WHERE id = $1`,
        [analysisId, totalRows, insertedRows.length, totalRows - insertedRows.length, JSON.stringify(output)]
      );
      await client.query('COMMIT');
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown CSV analysis failure';
    console.error(JSON.stringify({ analysisId, service: 'csv-analysis', error: message }));
    await pool.query(
      `UPDATE transaction_analyses
       SET status = 'failed',
           error_message = 'Analysis failed during processing. Check service health and upload the file again.',
           updated_at = NOW()
       WHERE id = $1`,
      [analysisId]
    );
  }
}
