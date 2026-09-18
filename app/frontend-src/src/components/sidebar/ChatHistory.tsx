import React from 'react';
import { Trash2 } from 'lucide-react';
import { useChatStore } from '../../store/chatStore';
import { useChat } from '../../hooks/useChat';
import { groupConversations, truncate } from '../../utils/formatters';
import { cn } from '../../utils/cn';

export default function ChatHistory() {
  const conversations = useChatStore((s) => s.conversations);
  const currentId = useChatStore((s) => s.currentConversationId);
  const { loadConversation, deleteConversation } = useChat();

  const groups = groupConversations(conversations);

  if (conversations.length === 0) {
    return (
      <p className="px-3 py-4 text-xs text-zinc-600 text-center">No conversations yet</p>
    );
  }

  return (
    <div className="flex flex-col gap-1 overflow-y-auto flex-1 min-h-0 px-2">
      {groups.map((group) => (
        <div key={group.label} className="mb-1">
          <p className="px-2 pt-3 pb-1 text-xs font-medium text-zinc-600 uppercase tracking-wider">
            {group.label}
          </p>
          {group.conversations.map((conv) => (
            <div
              key={conv.id}
              className={cn(
                'conversation-item group',
                currentId === conv.id && 'conversation-item--active',
              )}
              onClick={() => loadConversation(conv.id)}
            >
              <span className="flex-1 truncate text-sm">
                {truncate(conv.title || 'Untitled', 34)}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteConversation(conv.id);
                }}
                className="opacity-0 group-hover:opacity-100 transition-opacity text-zinc-500 hover:text-red-400 p-0.5 rounded"
                title="Delete"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
