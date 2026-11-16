import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  Calculation,
  ChatResponse,
  Citation,
  ConversationSummary,
  FINANCEBENCH_COMPANIES,
  getConversation,
  getEvaluationResults,
  listConversations,
  listDocuments,
  sendChat,
} from './api/client';

type UiMessage = {
  role: 'user' | 'assistant';
  content: string;
  payload?: ChatResponse;
};

function statusClass(status?: string) {
  if (!status) return '';
  if (status === 'answered') return 'status-ok';
  if (status === 'refused') return 'status-refuse';
  return 'status-warn';
}

export default function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [company, setCompany] = useState('3M');
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [filingCount, setFilingCount] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evalMetrics, setEvalMetrics] = useState<Record<string, number | null> | null>(null);
  const [showEval, setShowEval] = useState(false);

  const companies = useMemo(() => FINANCEBENCH_COMPANIES, []);

  async function refreshSidebar() {
    const [convos, docs] = await Promise.all([listConversations(), listDocuments()]);
    setConversations(convos);
    setFilingCount(docs.indexed_filing_count || docs.items.length);
    setChunkCount(docs.chunk_count || 0);
  }

  useEffect(() => {
    refreshSidebar().catch((err) => setError(String(err)));
  }, []);

  async function openConversation(id: string) {
    setActiveId(id);
    setSelectedCitation(null);
    const detail = await getConversation(id);
    const mapped: UiMessage[] = detail.messages.map((item) => ({
      role: item.role === 'assistant' ? 'assistant' : 'user',
      content: item.content,
      payload: item.payload,
    }));
    setMessages(mapped);
    const lastAssistant = [...mapped].reverse().find((item) => item.role === 'assistant' && item.payload?.citations?.length);
    if (lastAssistant?.payload?.citations?.[0]) {
      setSelectedCitation(lastAssistant.payload.citations[0]);
    }
  }

  function startNewChat() {
    setActiveId(null);
    setMessages([]);
    setSelectedCitation(null);
    setError(null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || loading) return;
    setLoading(true);
    setError(null);
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setDraft('');
    try {
      const response = await sendChat(text, activeId, company || null);
      setActiveId(response.conversation_id);
      setMessages((prev) => [...prev, { role: 'assistant', content: response.answer, payload: response }]);
      if (response.citations?.[0]) {
        setSelectedCitation(response.citations[0]);
      } else {
        setSelectedCitation(null);
      }
      await refreshSidebar();
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function loadEval() {
    setShowEval(true);
    try {
      const results = await getEvaluationResults();
      setEvalMetrics(results.items?.[0]?.metrics || null);
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="copilot-shell">
      <aside className="pane left-pane">
        <div className="brand">Finsight</div>
        <button className="primary-btn full" onClick={startNewChat}>
          New chat
        </button>
        <div className="pane-label">Conversations</div>
        <div className="conversation-list">
          {conversations.map((item) => (
            <button
              key={item.id}
              className={`conversation-item ${activeId === item.id ? 'active' : ''}`}
              onClick={() => openConversation(item.id).catch((err) => setError(String(err)))}
            >
              {item.title}
            </button>
          ))}
        </div>
        <button className="ghost-btn" onClick={() => loadEval().catch((err) => setError(String(err)))}>
          Evaluation
        </button>
      </aside>

      <main className="pane center-pane">
        <header className="copilot-header">
          <div>
            <h1>Financial Report Copilot</h1>
            <p>
              Indexed filings: {filingCount} · Chunks: {chunkCount}
            </p>
          </div>
          <label className="company-filter">
            Company
            <select value={company} onChange={(event) => setCompany(event.target.value)}>
              <option value="">All companies</option>
              {companies.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </header>

        {showEval && (
          <section className="eval-panel">
            <h2>Latest evaluation</h2>
            {evalMetrics ? (
              <ul>
                <li>Answer pass rate: {evalMetrics.answer_pass_rate ?? 'n/a'}</li>
                <li>Retrieval recall@5: {evalMetrics.mean_retrieval_recall_at_5 ?? 'n/a'}</li>
                <li>Numeric match rate: {evalMetrics.numeric_match_rate ?? 'n/a'}</li>
                <li>Refusal accuracy: {evalMetrics.refusal_accuracy ?? 'n/a'}</li>
              </ul>
            ) : (
              <p>No evaluation runs yet.</p>
            )}
          </section>
        )}

        <section className="message-list">
          {messages.length === 0 && (
            <div className="empty-state">
              Ask an extraction question about an indexed filing, or try a buy/sell prompt to see a refusal.
            </div>
          )}
          {messages.map((message, index) => (
            <div key={`${message.role}-${index}`} className={`message ${message.role}`}>
              <div className="message-role">{message.role}</div>
              {message.role === 'assistant' && message.payload ? (
                <AnswerCard
                  payload={message.payload}
                  onSelectCitation={setSelectedCitation}
                  selectedId={selectedCitation?.source_id}
                />
              ) : (
                <div className="message-body">{message.content}</div>
              )}
            </div>
          ))}
        </section>

        {error && <div className="error-banner">{error}</div>}

        <form className="composer" onSubmit={onSubmit}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask about FY metrics, margins, or filing disclosures…"
            rows={3}
          />
          <button className="primary-btn" disabled={loading || !draft.trim()}>
            {loading ? 'Running…' : 'Send'}
          </button>
        </form>
      </main>

      <aside className="pane right-pane">
        <div className="pane-label">Evidence</div>
        {!selectedCitation && (
          <div className="empty-state compact">Select a citation chip to inspect the source snippet, page, and period.</div>
        )}
        {selectedCitation && (
          <div className="evidence-card">
            <h2>{selectedCitation.document_name}</h2>
            <p className="meta">
              Section: {selectedCitation.section}
              <br />
              Page index: {selectedCitation.page_index} (display {selectedCitation.display_page})
              <br />
              Period: {selectedCitation.period || 'n/a'}
            </p>
            <pre className="snippet">{selectedCitation.snippet}</pre>
          </div>
        )}
      </aside>
    </div>
  );
}

function AnswerCard({
  payload,
  onSelectCitation,
  selectedId,
}: {
  payload: ChatResponse;
  onSelectCitation: (citation: Citation) => void;
  selectedId?: string;
}) {
  return (
    <div className={`answer-card ${statusClass(payload.status)}`}>
      <div className="answer-status">{payload.status}</div>
      {payload.refusal_reason && <div className="refusal-reason">Reason: {payload.refusal_reason}</div>}
      <div className="message-body">{payload.answer}</div>
      {payload.calculations?.length > 0 && (
        <div className="calc-block">
          <div className="pane-label">Calculations</div>
          {payload.calculations.map((calc: Calculation, index: number) => (
            <div key={index} className="calc-item">
              <div>
                {calc.op || 'metric'}: {String(calc.result)} {calc.units}
              </div>
              <div className="meta">
                {calc.formula} · inputs {JSON.stringify(calc.inputs)}
              </div>
            </div>
          ))}
        </div>
      )}
      {payload.citations?.length > 0 && (
        <div className="citation-row">
          {payload.citations.map((citation) => (
            <button
              key={citation.source_id}
              className={`citation-chip ${selectedId === citation.source_id ? 'active' : ''}`}
              onClick={() => onSelectCitation(citation)}
            >
              {citation.document_name} p.{citation.display_page}
            </button>
          ))}
        </div>
      )}
      {payload.status === 'provider_unavailable' && (
        <div className="meta">Provider unavailable. Check LLM endpoint configuration.</div>
      )}
      {payload.status === 'insufficient_evidence' && <div className="meta">Insufficient indexed evidence for a grounded answer.</div>}
    </div>
  );
}
