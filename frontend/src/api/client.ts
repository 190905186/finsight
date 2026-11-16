export type AnswerStatus =
  | 'answered'
  | 'refused'
  | 'clarification_required'
  | 'insufficient_evidence'
  | 'provider_unavailable';

export type Citation = {
  source_id: string;
  document_name: string;
  section: string;
  page_index: number;
  display_page: number;
  period?: string | null;
  snippet: string;
  citation?: string;
  score?: number;
};

export type Calculation = {
  formula: string;
  inputs: Record<string, unknown>;
  result: number | string;
  units: string;
  op?: string;
};

export type ChatResponse = {
  status: AnswerStatus;
  answer: string;
  citations: Citation[];
  calculations: Calculation[];
  basis: string;
  evidence_status: string;
  refusal_reason?: string | null;
  request_id: string;
  conversation_id: string;
  trace: Array<{ node: string; status: string; detail?: string }>;
};

export type ConversationSummary = {
  id: string;
  title: string;
  company_filter?: string | null;
  updated_at?: string | null;
};

const TOKEN_KEY = 'finsight_token';

function authHeaders(): HeadersInit {
  const token = localStorage.getItem(TOKEN_KEY);
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function ensureAuth(): Promise<void> {
  if (localStorage.getItem(TOKEN_KEY)) return;
  const response = await fetch('/api/v1/auth/login', { method: 'POST' });
  if (!response.ok) throw new Error('Login failed');
  const payload = await response.json();
  localStorage.setItem(TOKEN_KEY, payload.token);
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  await ensureAuth();
  const response = await fetch(path, {
    ...init,
    headers: {
      ...authHeaders(),
      ...(init?.headers || {}),
    },
  });
  if (response.status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    await ensureAuth();
    const retry = await fetch(path, {
      ...init,
      headers: {
        ...authHeaders(),
        ...(init?.headers || {}),
      },
    });
    if (!retry.ok) {
      throw new Error(await retry.text());
    }
    return retry.json();
  }
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const payload = await api<{ items: ConversationSummary[] }>('/api/v1/copilot/conversations');
  return payload.items;
}

export async function getConversation(id: string) {
  return api<{ id: string; title: string; messages: Array<{ role: string; content: string; payload?: ChatResponse }> }>(
    `/api/v1/copilot/conversations/${id}`,
  );
}

export async function sendChat(message: string, conversationId?: string | null, company?: string | null) {
  return api<ChatResponse>('/api/v1/copilot/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      conversation_id: conversationId || undefined,
      company: company || undefined,
    }),
  });
}

export async function listDocuments() {
  return api<{
    items: Array<{ id: number; name: string; company: string; type: string; period?: string }>;
    indexed_filing_count: number;
    chunk_count: number;
  }>('/api/v1/documents');
}

export async function listCompanies() {
  return api<{ items: string[] }>('/api/v1/documents/companies');
}

export async function getEvaluationResults() {
  return api<{ items: Array<{ id: string; status: string; metrics?: Record<string, number | null> }> }>(
    '/api/v1/evaluation/results',
  );
}

export const FINANCEBENCH_COMPANIES = [
  '3M',
  'AES Corporation',
  'AMD',
  'Activision Blizzard',
  'Adobe',
  'Amazon',
  'Amcor',
  'American Express',
  'American Water Works',
  'Best Buy',
  'Block',
  'Boeing',
  'CVS Health',
  'Coca-Cola',
  'Corning',
  'Costco',
  'Foot Locker',
  'General Mills',
  'JPMorgan',
  'Johnson & Johnson',
  'Kraft Heinz',
  'Lockheed Martin',
  'MGM Resorts',
  'Microsoft',
  'Netflix',
  'Nike',
  'Paypal',
  'PepsiCo',
  'Pfizer',
  'Ulta Beauty',
  'Verizon',
  'Walmart',
];
