import React, { useState } from 'react';
import { Search, Settings, User, ChevronLeft, ChevronRight, Shield } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useUIStore } from '../../store/uiStore';
import { useChatStore } from '../../store/chatStore';
import NewChatButton from './NewChatButton';
import ChatHistory from './ChatHistory';
import ThemeToggle from '../common/ThemeToggle';
import { cn } from '../../utils/cn';

export default function Sidebar() {
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const [searchQuery, setSearchQuery] = useState('');
  const conversations = useChatStore((s) => s.conversations);
  const navigate = useNavigate();

  // Filter conversations by search
  const filtered = searchQuery
    ? conversations.filter((c) =>
        c.title?.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : conversations;

  return (
    <>
      {/* Sidebar panel */}
      <aside
        className={cn(
          'sidebar',
          sidebarOpen ? 'sidebar--open' : 'sidebar--closed',
        )}
      >
        {/* Top section */}
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between px-3 pt-4 pb-3">
            <div className="flex items-center gap-2">
              <Shield size={16} className="text-indigo-400" />
              <span className="text-sm font-semibold text-zinc-200">SovereignAI</span>
            </div>
            <button
              onClick={toggleSidebar}
              className="sidebar-toggle-btn"
              title="Collapse sidebar"
            >
              <ChevronLeft size={15} />
            </button>
          </div>

          {/* New Chat */}
          <div className="px-2 pb-2">
            <NewChatButton />
          </div>

          {/* Search */}
          <div className="px-2 pb-2">
            <div className="relative">
              <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
              <input
                type="text"
                placeholder="Search conversations…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-zinc-800/60 border border-zinc-700/50 rounded-md pl-8 pr-3 py-1.5 text-sm text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/50 transition-colors"
              />
            </div>
          </div>

          {/* Conversation history */}
          <div className="flex-1 overflow-hidden">
            <ChatHistory />
          </div>

          {/* Bottom section */}
          <div className="border-t border-zinc-800 px-2 py-3 flex flex-col gap-1">
            <ThemeToggle />
            <button
              onClick={() => navigate('/settings')}
              className="sidebar-bottom-btn"
            >
              <Settings size={14} />
              <span>Settings</span>
            </button>
            <button className="sidebar-bottom-btn">
              <User size={14} />
              <span>Profile</span>
            </button>
          </div>
        </div>
      </aside>

      {/* Collapsed toggle button (shown when sidebar is closed) */}
      {!sidebarOpen && (
        <button
          onClick={toggleSidebar}
          className="fixed top-4 left-4 z-50 p-2 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors shadow-lg"
          title="Open sidebar"
        >
          <ChevronRight size={15} />
        </button>
      )}
    </>
  );
}
