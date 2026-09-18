import { apiClient } from './client';
import type {
  Conversation,
  Message,
  Model,
  SendMessageParams,
  SendMessageResponse,
} from '../types';

export const chatApi = {
  /** List all available models from the backend */
  async getModels(): Promise<Model[]> {
    const { data } = await apiClient.get<Model[]>('/api/models');
    return data;
  },

  /** List recent conversations (up to 50) */
  async getConversations(): Promise<Conversation[]> {
    const { data } = await apiClient.get<Conversation[]>('/api/conversations');
    return data;
  },

  /** Load all messages for a given conversation */
  async getConversationMessages(
    conversationId: string,
  ): Promise<Message[]> {
    const { data } = await apiClient.get<{ conversation_id: string; messages: Message[] }>(
      `/api/conversations/${conversationId}`,
    );
    return data.messages;
  },

  /** Delete a conversation and all its messages */
  async deleteConversation(conversationId: string): Promise<void> {
    await apiClient.delete(`/api/conversations/${conversationId}`);
  },

  /**
   * Send a chat message.
   * The backend returns the full reply synchronously (no streaming).
   * The frontend animates the text reveal after receiving the response.
   */
  async sendMessage(params: SendMessageParams): Promise<SendMessageResponse> {
    const { data } = await apiClient.post<SendMessageResponse>('/api/chat', {
      message: params.message,
      model: params.model || '',
      conversation_id: params.conversation_id || '',
      use_rag: params.use_rag ?? true,
    });
    return data;
  },

  /**
   * Run an agentic task (uses the /api/agent endpoint).
   * The backend processes the task with planning + tool use.
   */
  async runAgentTask(task: string): Promise<{ final_answer: string; task_id: string; status: string }> {
    const { data } = await apiClient.post('/api/agent', { task });
    return data;
  },

  /** Get backend status (offline check, registered models) */
  async getStatus(): Promise<{ status: string; offline: boolean; models: Model[]; rag_docs: number }> {
    const { data } = await apiClient.get('/api/status');
    return data;
  },
};
