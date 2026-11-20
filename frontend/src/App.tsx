import { useState, type FormEvent } from 'react';

type Citation = {
  source: string;
  document: string;
  page: number;
  snippet: string;
  url?: string | null;
};

type MetricInput = {
  name: string | null;
  value: string;
  unit: string;
  period?: string | null;
  source?: string | null;
};

type Metric = {
  key: string;
  label: string;
  value: string | null;
  unit: string;
  status: string;
  formula: string;
  inputs: MetricInput[];
  source?: string;
  source_urls?: string[];
};

type ChatResponse = {
  conversation_id: string;
  prompting_strategy: PromptingStrategy;
  status: string;
  answer: string;
  company?: string | null;
  fiscal_year?: number | null;
  metrics: Metric[];
  macro_context?: {
    label: string;
    value?: string | null;
    unit: string;
    period?: string | null;
    source?: string | null;
  } | null;
  citations: Citation[];
  data_status: Record<string, string>;
  agent_trace: string[];
};

type ChatTurn = {
  role: 'user' | 'assistant';
  content: string;
};

type PromptingStrategy = 'zero_shot' | 'few_shot' | 'cot';

const MAX_QUERY_LENGTH = 4000;
const API_BASE_URL = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '');

function isPromptingStrategy(value: string): value is PromptingStrategy {
  return value === 'zero_shot' || value === 'few_shot' || value === 'cot';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

async function getRequestError(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (isRecord(body) && typeof body.detail === 'string') {
      return body.detail;
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
  const body: unknown = await response.json();
  if (!isRecord(body) || typeof body.token !== 'string' || !body.token.trim()) {
    throw new Error('The authentication endpoint returned no access token.');
  }
  return body.token;
}

function formatMetric(metric: Metric): string {
  if (metric.value === null || metric.status !== 'computed') {
    return metric.status === 'not_computable' ? 'Not computable' : 'Not available';
  }
  if (metric.unit === '%') {
    return `${Number(metric.value).toLocaleString(undefined, {
      maximumFractionDigits: 2,
    })}%`;
  }
  if (metric.unit === 'x') {
    return `${Number(metric.value).toLocaleString(undefined, {
      maximumFractionDigits: 4,
    })}x`;
  }
  const amount = Number(metric.value).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
  if (
    ['million', 'millions', 'usd millions'].includes(metric.unit.toLowerCase())
  ) {
    return `$${amount} million`;
  }
  if (metric.unit.toLowerCase() === 'usd') {
    return `$${amount}`;
  }
  return `${amount} ${metric.unit}`;
}

type AgentStepInfo = {
  title: string;
  status: string;
  detail: string;
};

function getAgentStepInfo(agent: string, report: ChatResponse): AgentStepInfo {
  const citationCount = report.citations.length;
  const computedMetrics = report.metrics.filter(
    (metric) => metric.status === 'computed' && metric.value !== null,
  ).length;

  switch (agent) {
    case 'plan':
      return {
        title: 'Request planning',
        status: 'completed',
        detail: [
          report.company || 'Company not specified',
          report.fiscal_year ? `FY${report.fiscal_year}` : null,
        ]
          .filter(Boolean)
          .join(' · '),
      };
    case 'rag_researcher': {
      const knowledgeStatus = report.data_status.knowledge_base ?? 'unknown';
      const documents = [...new Set(report.citations.map((item) => item.document))];
      return {
        title: 'Filing evidence retrieval',
        status: knowledgeStatus,
        detail: citationCount
          ? `${citationCount} excerpts from ${documents.join(', ')}.`
          : 'No filing excerpts were returned.',
      };
    }
    case 'public_data_researcher': {
      const sources = [
        ['SEC company facts', report.data_status.sec_companyfacts],
        ['SEC prior year', report.data_status.sec_prior_year],
        ['Treasury context', report.data_status.treasury_context],
      ].filter((source): source is [string, string] => typeof source[1] === 'string');
      const ready = sources.every(
        ([, status]) => status === 'not_requested' || status.startsWith('ready'),
      );
      return {
        title: 'Public data research',
        status: sources.length ? (ready ? 'ready' : 'review') : 'not_requested',
        detail: sources.length
          ? sources.map(([name, status]) => `${name}: ${status.replace(/_/g, ' ')}`).join(' · ')
          : 'No public data source was requested.',
      };
    }
    case 'fact_extractor': {
      const status = report.data_status.fact_extraction ?? 'unknown';
      return {
        title: 'Financial fact extraction',
        status,
        detail: `Evidence extraction status: ${status.replace(/_/g, ' ')}.`,
      };
    }
    case 'metric_calculator':
      return {
        title: 'Deterministic metric calculation',
        status: 'completed',
        detail: `${computedMetrics} of ${report.metrics.length} metrics computed from available facts.`,
      };
    case 'financial_analyst': {
      const status = report.data_status.financial_analyst ?? 'completed';
      return {
        title: 'Answer synthesis',
        status,
        detail:
          status === 'rate_limited_evidence_fallback'
            ? 'The model was rate-limited; the response uses retrieved filing excerpts.'
            : 'Assembled the response from available report data and filing evidence.',
      };
    }
    case 'report_verifier':
      return {
        title: 'Citation and answer verification',
        status: report.status,
        detail: `${citationCount} filing excerpts attached; the answer passed citation checks.`,
      };
    default:
      return {
        title: agent.replace(/_/g, ' '),
        status: 'completed',
        detail: 'Agent step completed.',
      };
  }
}

export default function App() {
  const [company, setCompany] = useState('3M');
  const [fiscalYear, setFiscalYear] = useState('2018');
  const [promptingStrategy, setPromptingStrategy] =
    useState<PromptingStrategy>('zero_shot');
  const [query, setQuery] = useState(
    'Prepare a cited financial report. Include revenue, margins, liquidity, leverage, and free cash flow where supported.',
  );
  const [conversationId, setConversationId] = useState<string>(() => crypto.randomUUID());
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<ChatResponse | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const normalizedQuery = query.trim();
    if (!normalizedQuery) {
      setError('Enter a question or report request.');
      return;
    }
    if (normalizedQuery.length > MAX_QUERY_LENGTH) {
      setError(`Keep the question to ${MAX_QUERY_LENGTH} characters or fewer.`);
      return;
    }

    setIsLoading(true);
    try {
      const token = accessToken ?? (await requestAccessToken());
      if (!accessToken) setAccessToken(token);

      const response = await fetch(`${API_BASE_URL}/api/v1/copilot/chat`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: normalizedQuery,
          conversation_id: conversationId,
          company: company.trim() || undefined,
          fiscal_year: Number(fiscalYear) || undefined,
          prompting_strategy: promptingStrategy,
        }),
      });
      if (!response.ok) throw new Error(await getRequestError(response));

      const body: unknown = await response.json();
      if (
        !isRecord(body) ||
        typeof body.answer !== 'string' ||
        !Array.isArray(body.metrics) ||
        !Array.isArray(body.citations) ||
        typeof body.conversation_id !== 'string' ||
        typeof body.prompting_strategy !== 'string' ||
        !isPromptingStrategy(body.prompting_strategy)
      ) {
        throw new Error('The copilot returned an unexpected response.');
      }

      const result = body as unknown as ChatResponse;
      setReport(result);
      setConversationId(result.conversation_id);
      setTurns((previous) => [
        ...previous,
        { role: 'user', content: normalizedQuery },
        { role: 'assistant', content: result.answer },
      ]);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'The report request could not be completed.',
      );
    } finally {
      setIsLoading(false);
    }
  }

  function startNewConversation() {
    setConversationId(crypto.randomUUID());
    setTurns([]);
    setReport(null);
    setError(null);
  }

  const displayedMetrics = report?.metrics ?? [];
  const hasComputedMetrics = displayedMetrics.some(
    (metric) => metric.status === 'computed' && metric.value !== null,
  );

  return (
    <div className="page-shell">
      <aside className="sidebar">
        <div className="sidebar-header">Finsight</div>
        <p className="sidebar-caption">Financial research workspace</p>
        <nav aria-label="Main navigation">
          <a className="active" href="#dashboard">Report dashboard</a>
          <a href="#analyst-chat">Analyst chat</a>
          <a href="#sources">Evidence and sources</a>
        </nav>
        <div className="sidebar-note">
          Reports are decision support. Verify source filings before relying on a
          result.
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">RAG · LangGraph · MCP</p>
            <h1>Financial report copilot</h1>
            <p>Evidence-backed metrics, source filings, and follow-up chat.</p>
          </div>
          <button className="secondary-btn" type="button" onClick={startNewConversation}>
            New conversation
          </button>
        </header>

        <section className="panel request-panel" id="dashboard">
          <div className="panel-heading">
            <div>
              <h2>Build a financial report</h2>
              <p>
                Choose an issuer and fiscal year. The report combines indexed
                filings, public-data tools, and deterministic calculations.
              </p>
            </div>
            <span className={`status-pill ${report?.status ?? 'idle'}`}>
              {report?.status ?? 'Ready'}
            </span>
          </div>
          <form className="report-form" onSubmit={handleSubmit}>
            <div className="form-fields">
              <label>
                Company or ticker
                <input
                  value={company}
                  onChange={(event) => setCompany(event.target.value)}
                  placeholder="3M or MMM"
                  maxLength={255}
                />
              </label>
              <label>
                Fiscal year
                <input
                  type="number"
                  min="1990"
                  max="2100"
                  value={fiscalYear}
                  onChange={(event) => setFiscalYear(event.target.value)}
                />
              </label>
            </div>
            <label>
              Prompting strategy
              <select
                value={promptingStrategy}
                onChange={(event) =>
                  isPromptingStrategy(event.target.value) &&
                  setPromptingStrategy(event.target.value)
                }
              >
                <option value="zero_shot">Zero-shot</option>
                <option value="few_shot">Few-shot</option>
                <option value="cot">Chain-of-thought (private)</option>
              </select>
              <span className="field-help">
                Few-shot adds examples. Chain-of-thought guides internal analysis;
                only the concise, cited answer is shown.
              </span>
            </label>
            <label>
              Report request or follow-up
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                maxLength={MAX_QUERY_LENGTH}
                required
              />
            </label>
            <div className="form-footer">
              <span>{query.length}/{MAX_QUERY_LENGTH}</span>
              <button className="primary-btn" type="submit" disabled={isLoading}>
                {isLoading ? 'Researching sources…' : 'Generate report'}
              </button>
            </div>
          </form>
          {error && <p className="error-message" role="alert">{error}</p>}
        </section>

        {report && (
          <>
            <section className="metrics-section" aria-live="polite">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Verified calculations</p>
                  <h2>
                    {report.company || company || 'Financial report'}{' '}
                    {report.fiscal_year ? `· FY${report.fiscal_year}` : ''}
                  </h2>
                </div>
                {report.prompting_strategy && (
                  <span className="status-pill">
                    {report.prompting_strategy.replace('_', ' ')} prompting
                  </span>
                )}
              </div>
              <p className="evidence-summary" role="status">
                {report.citations.length > 0 ? (
                  <>
                    {report.citations.length} filing excerpts cited.{' '}
                    <a href="#sources">View evidence and cited details</a>
                  </>
                ) : (
                  'No filing excerpts were returned for this request.'
                )}
              </p>
              {displayedMetrics.length > 0 && !hasComputedMetrics && (
                <p className="metric-warning" role="status">
                  No financial metric could be verified from the current filing
                  or public data sources. Review the Research pipeline statuses
                  below for the missing source or service.
                </p>
              )}
              <div className="metrics-grid">
                {displayedMetrics.map((metric) => (
                  <article className="metric-card" key={metric.key}>
                    <div className="metric-label">{metric.label}</div>
                    <div className="metric-value">{formatMetric(metric)}</div>
                    <div className={`metric-status ${metric.status}`}>
                      {metric.status.replace(/_/g, ' ')}
                    </div>
                    <details>
                      <summary>Formula and inputs</summary>
                      <p className="formula">{metric.formula}</p>
                      {metric.inputs.map((input, index) => (
                        <p className="metric-input" key={`${input.name}-${index}`}>
                          {input.name}: {input.value} {input.unit}
                          {input.period ? ` · ${input.period}` : ''}
                          {input.source ? ` · ${input.source}` : ''}
                        </p>
                      ))}
                      {metric.source_urls?.map((url) => (
                        <a
                          className="source-link"
                          href={url}
                          key={url}
                          rel="noreferrer"
                          target="_blank"
                        >
                          Open primary data source
                        </a>
                      ))}
                    </details>
                  </article>
                ))}
              </div>
              {report.macro_context && (
                <p className="macro-context">
                  Public context — {report.macro_context.label}:{' '}
                  {report.macro_context.value} {report.macro_context.unit}
                  {report.macro_context.period
                    ? ` (${report.macro_context.period})`
                    : ''}
                  {report.macro_context.source && (
                    <>
                      {' · '}
                      <a href={report.macro_context.source} rel="noreferrer" target="_blank">
                        source
                      </a>
                    </>
                  )}
                </p>
              )}
            </section>

            <div className="panel-grid">
              <section className="panel analyst-panel" id="analyst-chat">
                <div className="panel-heading">
                  <div>
                    <h2>Analyst conversation</h2>
                    <p>Ask follow-ups in this thread; short-term context is retained.</p>
                  </div>
                </div>
                <div className="chat-history" aria-live="polite">
                  {turns.map((turn, index) => (
                    <article className={`chat-turn ${turn.role}`} key={index}>
                      <strong>{turn.role === 'user' ? 'You' : 'Finsight analyst'}</strong>
                      <p>{turn.content}</p>
                    </article>
                  ))}
                </div>
              </section>

              <section className="panel data-status-panel">
                <h2>Research pipeline</h2>
                {Object.entries(report.data_status).map(([source, state]) => (
                  <div className="source-status" key={source}>
                    <span>{source.replace(/_/g, ' ')}</span>
                    <strong className={state.startsWith('ready') ? 'ok' : 'warning'}>
                      {state.replace(/_/g, ' ')}
                    </strong>
                  </div>
                ))}
              </section>
            </div>

            {report.agent_trace.length > 0 && (
              <section className="panel pipeline-panel" aria-labelledby="pipeline-heading">
                <div className="panel-heading">
                  <div>
                    <h2 id="pipeline-heading">Agent execution pipeline</h2>
                    <p>
                      Ordered steps run for this request, with source status and
                      output details.
                    </p>
                  </div>
                </div>
                <ol className="agent-pipeline">
                  {report.agent_trace.map((agent, index) => {
                    const step = getAgentStepInfo(agent, report);
                    const statusClass =
                      step.status === 'completed' ||
                      step.status === 'ready' ||
                      step.status === 'not_requested' ||
                      step.status.startsWith('ready')
                        ? 'ok'
                        : 'warning';
                    return (
                      <li className="agent-step" key={`${agent}-${index}`}>
                        <span className="agent-step-index">{index + 1}</span>
                        <div className="agent-step-content">
                          <div className="agent-step-heading">
                            <h3>{step.title}</h3>
                            <span className={`agent-step-status ${statusClass}`}>
                              {step.status.replace(/_/g, ' ')}
                            </span>
                          </div>
                          <p>{step.detail}</p>
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </section>
            )}

            <section className="panel sources-panel" id="sources">
              <h2>Evidence and citations</h2>
              {report.citations.length ? (
                <div className="citation-list">
                  {report.citations.map((citation, index) => (
                    <article
                      className="citation-card"
                      key={`${citation.source}-${citation.document}-${citation.page}-${index}`}
                    >
                      <div className="citation-title">
                        <strong>{citation.source}</strong>
                        <span>{citation.document} · PDF page {citation.page}</span>
                      </div>
                      <p>{citation.snippet}</p>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="empty-state">
                  No indexed filing evidence was returned for this request.
                </p>
              )}
            </section>
          </>
        )}

        {!report && (
          <section className="empty-dashboard">
            <div className="empty-icon">F</div>
            <h2>Your report will appear here</h2>
            <p>
              Start with the sample query for 3M FY2018, or choose another
              company and year from the indexed reference corpus.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
