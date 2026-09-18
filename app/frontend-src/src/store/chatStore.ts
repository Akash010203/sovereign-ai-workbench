import { create } from 'zustand';
import type { Conversation, Message, Attachment, Model, AgentMode } from '../types';

interface ChatState {
  // Conversations
  conversations: Conversation[];
  currentConversationId: string | null;

  // Messages keyed by conversation ID
  messages: Record<string, Message[]>;

  // Streaming / loading
  isLoading: boolean;
  streamingMessageId: string | null;

  // Models
  models: Model[];
  selectedModel: string;

  // Agent mode
  agentMode: AgentMode;

  // Pending attachments (before send)
  attachments: Attachment[];

  // ── Actions ──────────────────────────────────────────────────────────────────

  setConversations: (convs: Conversation[]) => void;
  prependConversation: (conv: Conversation) => void;
  updateConversationTitle: (id: string, title: string) => void;
  removeConversation: (id: string) => void;

  setCurrentConversationId: (id: string | null) => void;

  getMessages: (convId: string) => Message[];
  setMessages: (convId: string, msgs: Message[]) => void;
  appendMessage: (convId: string, msg: Message) => void;
  updateMessage: (convId: string, msgId: string, update: Partial<Message>) => void;

  setLoading: (v: boolean) => void;
  setStreamingMessageId: (id: string | null) => void;

  setModels: (models: Model[]) => void;
  setSelectedModel: (name: string) => void;

  setAgentMode: (mode: AgentMode) => void;

  addAttachment: (a: Attachment) => void;
  updateAttachment: (id: string, update: Partial<Attachment>) => void;
  removeAttachment: (id: string) => void;
  clearAttachments: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  currentConversationId: null,
  messages: {},
  isLoading: false,
  streamingMessageId: null,
  models: [],
  selectedModel: '',
  agentMode: 'auto',
  attachments: [],

  // Conversations
  setConversations: (convs) => set({ conversations: convs }),
  prependConversation: (conv) =>
    set((s) => ({ conversations: [conv, ...s.conversations] })),
  updateConversationTitle: (id, title) =>
    set((s) => ({
      conversations: s.conversations.map((c) =>
        c.id === id ? { ...c, title } : c,
      ),
    })),
  removeConversation: (id) =>
    set((s) => ({
      conversations: s.conversations.filter((c) => c.id !== id),
      currentConversationId:
        s.currentConversationId === id ? null : s.currentConversationId,
      messages: Object.fromEntries(
        Object.entries(s.messages).filter(([k]) => k !== id),
      ),
    })),

  setCurrentConversationId: (id) => set({ currentConversationId: id }),

  // Messages
  getMessages: (convId) => get().messages[convId] ?? [],
  setMessages: (convId, msgs) =>
    set((s) => ({ messages: { ...s.messages, [convId]: msgs } })),
  appendMessage: (convId, msg) =>
    set((s) => ({
      messages: {
        ...s.messages,
        [convId]: [...(s.messages[convId] ?? []), msg],
      },
    })),
  updateMessage: (convId, msgId, update) =>
    set((s) => ({
      messages: {
        ...s.messages,
        [convId]: (s.messages[convId] ?? []).map((m) =>
          m.id === msgId ? { ...m, ...update } : m,
        ),
      },
    })),

  // Loading / streaming
  setLoading: (v) => set({ isLoading: v }),
  setStreamingMessageId: (id) => set({ streamingMessageId: id }),

  // Models
  setModels: (models) => set({ models }),
  setSelectedModel: (name) => set({ selectedModel: name }),

  // Mode
  setAgentMode: (mode) => set({ agentMode: mode }),

  // Attachments
  addAttachment: (a) => set((s) => ({ attachments: [...s.attachments, a] })),
  updateAttachment: (id, update) =>
    set((s) => ({
      attachments: s.attachments.map((a) =>
        a.id === id ? { ...a, ...update } : a,
      ),
    })),
  removeAttachment: (id) =>
    set((s) => ({ attachments: s.attachments.filter((a) => a.id !== id) })),
  clearAttachments: () => set({ attachments: [] }),
}));
