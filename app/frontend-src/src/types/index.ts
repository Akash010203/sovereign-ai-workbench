// ─── Core entities ────────────────────────────────────────────────────────────

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  model_used?: string;
  sources?: RAGSource[];
  attachments?: AttachmentMeta[];
}

export interface Conversation {
  id: string;
  title: string;
  model_name?: string;
  created_at: string;
  updated_at: string;
}

export interface RAGSource {
  source: string;
  chunk_id?: string;
  score?: number;
}

export interface Model {
  name: string;
  model_type: string;
  available: boolean;
}

// ─── Attachments ──────────────────────────────────────────────────────────────

export type AttachmentStatus = 'pending' | 'reading' | 'uploading' | 'done' | 'error';

export interface Attachment {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
  status: AttachmentStatus;
  progress: number;
  error?: string;
  /** extracted text content once read client-side */
  textContent?: string;
}

/** Serialisable metadata saved alongside a message (no File object) */
export interface AttachmentMeta {
  id: string;
  name: string;
  size: number;
  type: string;
}

// ─── Chat API request/response shapes ────────────────────────────────────────

export interface SendMessageParams {
  message: string;
  model?: string;
  conversation_id?: string;
  use_rag?: boolean;
}

export interface SendMessageResponse {
  reply: string;
  conversation_id: string;
  model_used: string;
  category: string;
  routing_reason: string;
  rag_used: boolean;
  sources: RAGSource[];
}

// ─── UI ───────────────────────────────────────────────────────────────────────

export type AgentMode = 'auto' | 'research' | 'document' | 'vision' | 'reasoning';

export type Theme = 'dark' | 'light';

export interface ConversationGroup {
  label: string;
  conversations: Conversation[];
}
