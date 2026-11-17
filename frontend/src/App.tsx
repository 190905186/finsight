import { useState, type FormEvent } from 'react';

type MetricCardProps = {
  label: string;
  value: string;
  delta: string;
};

type RAGCitation = {
  source: string;
  document: string;
  page: number;
  snippet: string;
};

type RAGResponse = {
  answer: string;
  citations: RAGCitation[];
};

const MAX_QUERY_LENGTH = 4000;
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isRAGCitation(value: unknown): value is RAGCitation {
  return (
    isRecord(value) &&
    typeof value.source === 'string' &&
    typeof value.document === 'string' &&
    typeof value.page === 'number' &&
    Number.isInteger(value.page) &&
    value.page > 0 &&
    typeof value.snippet === 'string'
  );
}

function isRAGResponse(value: unknown): value is RAGResponse {
  return (
    isRecord(value) &&
    typeof value.answer === 'string' &&
    Array.isArray(value.citations) &&
    value.citations.every(isRAGCitation)
  );
}

async function getRequestError(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (isRecord(body)) {
      if (typeof body.detail === 'string' && body.detail.trim()) {
        return body.detail;
      }
      if (Array.isArray(body.detail)) {
        const messages = body.detail.flatMap((item) =>
          isRecord(item) && typeof item.msg === 'string' ? [item.msg] : [],
        );
        if (messages.length > 0) {
          return messages.join('; ');
        }
      }
    }
  } catch {
    return `Request failed with status ${response.status}.`;
  }
  return `Request failed with status ${response.status}.`;
}

async function requestAccessToken(): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(await getRequestError(response));
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error('The authentication endpoint returned invalid JSON.');
  }
  if (!isRecord(body) || typeof body.token !== 'string' || !body.token.trim()) {
    throw new Error('The authentication endpoint returned no access token.');
  }
  return body.token;
}

function MetricCard({ label, value, delta }: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-delta">{delta}</div>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState('');
  const [company, setCompany] = useState('');
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RAGResponse | null>(null);

  const metrics = [
    { label: 'Invoices processed', value: '1,248', delta: '+8.4% vs last month' },
    { label: 'Pending exceptions', value: '36', delta: '-12.1%' },
    { label: 'Duplicate invoices', value: '5', delta: '2 flagged for review' },
    { label: 'Expense violations', value: '11', delta: '+3 this week' },
  ];

  const rows = [
    { vendor: 'ABC Technologies', risk: 'Moderate', status: 'Manual review' },
    { vendor: 'Northwind Office Supply', risk: 'Low', status: 'Approved' },
    { vendor: 'BluePeak Logistics', risk: 'High', status: 'Escalated' },
  ];

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);

    const normalizedQuery = query.trim();
    if (!normalizedQuery) {
      setError('Enter a question about the financial reports.');
      return;
    }
    if (normalizedQuery.length > MAX_QUERY_LENGTH) {
      setError(`Keep the question to ${MAX_QUERY_LENGTH} characters or fewer.`);
      return;
    }

    setIsLoading(true);
    try {
      const token = accessToken ?? (await requestAccessToken());
      if (!accessToken) {
        setAccessToken(token);
      }

      const response = await fetch(`${API_BASE_URL}/api/v1/rag/query`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: normalizedQuery,
          top_k: 5,
          company: company.trim() || undefined,
        }),
      });
      if (!response.ok) {
        throw new Error(await getRequestError(response));
      }

      let body: unknown;
      try {
        body = await response.json();
      } catch {
        throw new Error('The RAG endpoint returned invalid JSON.');
      }
      if (!isRAGResponse(body)) {
        throw new Error('The RAG endpoint returned an unexpected response.');
      }
      setResult(body);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'The RAG request could not be completed.',
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <aside className="sidebar">
        <div className="sidebar-header">FinanceOps AI</div>
        <nav>
          <a href="#">Executive Dashboard</a>
          <a href="#">Invoices</a>
          <a href="#">Expenses</a>
          <a href="#">Vendor Risk</a>
          <a href="#">Anomalies</a>
          <a href="#">Audit</a>
        </nav>
      </aside>
      <main className="content">
        <header className="topbar">
          <div>
            <h1>Operations Copilot</h1>
            <p>Financial performance and control monitoring</p>
          </div>
          <button className="primary-btn">Run audit review</button>
        </header>

        <section className="metrics-grid">
          {metrics.map((metric) => (
            <MetricCard key={metric.label} {...metric} />
          ))}
        </section>

        <section className="panel-grid">
          <div className="panel rag-panel">
            <h2>Ask your financial reports</h2>
            <p className="rag-description">
              Ask a question about the indexed filings. Answers include source and PDF-page
              citations.
            </p>
            <form className="rag-form" onSubmit={handleSubmit}>
              <label htmlFor="rag-query">Question</label>
              <textarea
                id="rag-query"
                name="query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="For example: What were 3M's FY2018 purchases of property, plant and equipment?"
                maxLength={MAX_QUERY_LENGTH}
                required
                aria-describedby="rag-query-help"
              />
              <label htmlFor="rag-company">Company filter (optional)</label>
              <input
                id="rag-company"
                name="company"
                type="text"
                value={company}
                onChange={(event) => setCompany(event.target.value)}
                placeholder="For example: 3M"
                maxLength={255}
              />
              <div className="rag-form-footer">
                <span id="rag-query-help">
                  {query.length}/{MAX_QUERY_LENGTH} characters
                </span>
                <button className="primary-btn" type="submit" disabled={isLoading}>
                  {isLoading ? 'Searching reports…' : 'Ask RAG'}
                </button>
              </div>
            </form>

            {error && (
              <p className="rag-error" role="alert">
                {error}
              </p>
            )}

            {isLoading && (
              <p className="rag-loading" role="status">
                Searching indexed reports and preparing a cited answer…
              </p>
            )}

            {result && (
              <section className="rag-result" aria-live="polite">
                <h3>Response</h3>
                <p className="rag-answer">{result.answer}</p>
                {result.citations.length > 0 ? (
                  <div className="citation-list">
                    <h4>Sources</h4>
                    {result.citations.map((citation, index) => (
                      <article
                        className="citation-card"
                        key={`${citation.source}-${citation.document}-${citation.page}-${index}`}
                      >
                        <div className="citation-title">
                          <strong>{citation.source}</strong>
                          <span>
                            {citation.document} · PDF page {citation.page}
                          </span>
                        </div>
                        <p>{citation.snippet}</p>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="no-citations">
                    No supporting citation was returned. Treat this as an abstention.
                  </p>
                )}
              </section>
            )}
          </div>

          <div className="panel">
            <h2>Risk watchlist</h2>
            <table>
              <thead>
                <tr>
                  <th>Vendor</th>
                  <th>Risk</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.vendor}>
                    <td>{row.vendor}</td>
                    <td>{row.risk}</td>
                    <td>{row.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
