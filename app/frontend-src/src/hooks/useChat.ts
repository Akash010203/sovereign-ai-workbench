import { useCallback } from 'react';
import { chatApi } from '../api/chat';
import { useChatStore } from '../store/chatStore';
import type { Message, SendMessageParams, AttachmentMeta } from '../types';
import { generateId } from '../utils/fileUtils';

export function useChat() {
  const store = useChatStore();

  /**
   * Send a user message (and optional attached file context) to the backend.
   * Optimistically appends the user message, then awaits the AI reply.
   */
  const sendMessage = useCallback(
    async (
      userText: string,
      attachmentMetas: AttachmentMeta[] = [],
    ) => {
      // Build full message text — append file context hint if files are attached
      let fullText = userText.trim();
      if (attachmentMetas.length > 0) {
        const fileList = attachmentMetas.map((a) => a.name).join(', ');
        fullText = `${fullText}\n\n[Attached files: ${fileList}]`;
      }

      const currentConvId = store.currentConversationId;

      // Optimistic user message
      const userMsg: Message = {
        id: generateId(),
        role: 'user',
        content: userText,
        created_at: new Date().toISOString(),
        attachments: attachmentMetas.length > 0 ? attachmentMetas : undefined,
      };

      // Placeholder AI message (streaming state)
      const aiPlaceholderId = generateId();
      const aiPlaceholder: Message = {
        id: aiPlaceholderId,
        role: 'assistant',
        content: '',
        created_at: new Date().toISOString(),
      };

      // Append to whichever conv we know about now (may be null before first message)
      const convId = currentConvId ?? '';
      if (convId) {
        store.appendMessage(convId, userMsg);
        store.appendMessage(convId, aiPlaceholder);
      }

      store.setLoading(true);
      store.setStreamingMessageId(aiPlaceholderId);

      try {
        const params: SendMessageParams = {
          message: fullText,
          model: store.selectedModel || undefined,
          conversation_id: convId || undefined,
          use_rag: true,
        };

        const resp = await chatApi.sendMessage(params);

        // If this was a new conversation, we now have an ID
        const resolvedConvId = resp.conversation_id;

        if (!convId) {
          // First message — set current conv and add messages
          store.setCurrentConversationId(resolvedConvId);
          store.setMessages(resolvedConvId, [userMsg, { ...aiPlaceholder, content: '' }]);
          // Add a placeholder conversation to sidebar immediately
          store.prependConversation({
            id: resolvedConvId,
            title: userText.slice(0, 60),
            model_name: resp.model_used,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
        }

        // Update AI message with real content
        store.updateMessage(resolvedConvId, aiPlaceholderId, {
          content: resp.reply,
          model_used: resp.model_used,
          sources: resp.sources,
        });
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'An error occurred';
        const resolvedConvId = convId || store.currentConversationId || '';
        if (resolvedConvId) {
          store.updateMessage(resolvedConvId, aiPlaceholderId, {
            content: `⚠️ Error: ${errorMsg}`,
          });
        }
      } finally {
        store.setLoading(false);
        store.setStreamingMessageId(null);
        store.clearAttachments();
      }
    },
    [store],
  );

  /** Load conversation messages from the backend */
  const loadConversation = useCallback(
    async (conversationId: string) => {
      store.setCurrentConversationId(conversationId);
      // Only fetch if not already cached
      if (store.messages[conversationId]?.length) return;
      try {
        const msgs = await chatApi.getConversationMessages(conversationId);
        store.setMessages(conversationId, msgs);
      } catch (err) {
        console.error('Failed to load conversation:', err);
      }
    },
    [store],
  );

  /** Start a fresh new chat */
  const newChat = useCallback(() => {
    store.setCurrentConversationId(null);
    store.clearAttachments();
  }, [store]);

  /** Delete a conversation */
  const deleteConversation = useCallback(
    async (id: string) => {
      try {
        await chatApi.deleteConversation(id);
        store.removeConversation(id);
      } catch (err) {
        console.error('Failed to delete conversation:', err);
      }
    },
    [store],
  );

  return { sendMessage, loadConversation, newChat, deleteConversation };
}
