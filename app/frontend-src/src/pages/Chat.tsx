import React, { useEffect } from 'react';
import { useChatStore } from '../store/chatStore';
import { useUIStore } from '../store/uiStore';
import { chatApi } from '../api/chat';
import Sidebar from '../components/sidebar/Sidebar';
import ChatWindow from '../components/chat/ChatWindow';
import ModelSelector from '../components/common/ModelSelector';
import { truncate } from '../utils/formatters';
import { cn } from '../utils/cn';

export default function Chat() {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const currentConvId = useChatStore((s) => s.currentConversationId);
  const conversations = useChatStore((s) => s.conversations);
  const setConversations = useChatStore((s) => s.setConversations);

  const currentConv = conversations.find((c) => c.id === currentConvId);
  const pageTitle = currentConv?.title
    ? truncate(currentConv.title, 40)
    : 'New Conversation';

  // Load conversation list on mount
  useEffect(() => {
    chatApi.getConversations().then(setConversations).catch(() => {});
  }, []);

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <main className={cn('main-area', sidebarOpen ? 'main-area--sidebar-open' : '')}>
        {/* Header */}
        <header className="app-header">
          <div className="flex items-center gap-3 min-w-0">
            <h2 className="text-sm font-medium text-zinc-300 truncate">{pageTitle}</h2>
          </div>
          <div className="flex items-center gap-2">
            <ModelSelector />
          </div>
        </header>

        {/* Chat area */}
        <ChatWindow />
      </main>
    </div>
  );
}
