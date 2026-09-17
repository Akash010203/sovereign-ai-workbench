import React, { useEffect, useRef, useCallback } from 'react';
import { useChatStore } from '../../store/chatStore';
import { useChat } from '../../hooks/useChat';
import { useFileUpload } from '../../hooks/useFileUpload';
import Message from './Message';
import Composer from './Composer';
import DragDropOverlay from '../common/DragDropOverlay';
import { FileText, Cpu, BookOpen, AlertTriangle, Layers, Key } from 'lucide-react';

const SUGGESTIONS = [
  { icon: FileText, label: 'Summarise this document', prompt: 'Please summarise the key points of the attached document.' },
  { icon: Cpu, label: 'Analyse an engineering report', prompt: 'Analyse this engineering report and highlight critical findings.' },
  { icon: BookOpen, label: 'Explain this P&ID', prompt: 'Explain the components and flow in this P&ID diagram.' },
  { icon: AlertTriangle, label: 'Find risks in this report', prompt: 'Identify potential risks and safety concerns in this report.' },
  { icon: Layers, label: 'Compare two documents', prompt: 'Compare the two attached documents and summarise the differences.' },
  { icon: Key, label: 'Extract key information', prompt: 'Extract and list all key data points, dates, and figures from this document.' },
];

export default function ChatWindow() {
  const currentConvId = useChatStore((s) => s.currentConversationId);
  const messagesMap = useChatStore((s) => s.messages);
  const messages = currentConvId ? (messagesMap[currentConvId] ?? []) : [];
  const isLoading = useChatStore((s) => s.isLoading);
  const streamingMessageId = useChatStore((s) => s.streamingMessageId);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { addFiles } = useFileUpload();
  const { sendMessage } = useChat();

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSuggestion = useCallback(
    (prompt: string) => {
      sendMessage(prompt, []);
    },
    [sendMessage],
  );

  const isEmpty = !currentConvId || messages.length === 0;

  return (
    <div className="chat-window">
      <DragDropOverlay onDrop={addFiles} />

      {/* Messages area */}
      <div className="messages-area">
        {isEmpty ? (
          /* ── Empty / Landing state ─────────────────────────────────── */
          <div className="empty-state">
            <div className="empty-state__logo">
              <span className="text-3xl">⚡</span>
            </div>
            <h1 className="empty-state__heading">How can I help you today?</h1>
            <p className="empty-state__sub">
              Ask questions, analyse confidential documents, or work with your AI agents.
              <br />
              Everything stays on your device.
            </p>

            <div className="suggestions-grid">
              {SUGGESTIONS.map(({ icon: Icon, label, prompt }) => (
                <button
                  key={label}
                  className="suggestion-card"
                  onClick={() => handleSuggestion(prompt)}
                >
                  <Icon size={15} className="text-indigo-400 flex-shrink-0 mt-0.5" />
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* ── Messages list ─────────────────────────────────────────── */
          <div className="messages-list">
            {messages.map((msg) => (
              <div key={msg.id} className="animate-fade-in">
                <Message
                  message={msg}
                  isStreaming={msg.id === streamingMessageId}
                />
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Composer — always pinned to bottom */}
      <div className="composer-area">
        <Composer />
      </div>
    </div>
  );
}
