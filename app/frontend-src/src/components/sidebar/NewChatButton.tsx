import React from 'react';
import { Plus } from 'lucide-react';
import { useChat } from '../../hooks/useChat';

export default function NewChatButton() {
  const { newChat } = useChat();
  return (
    <button onClick={newChat} className="new-chat-btn">
      <Plus size={15} />
      <span>New Chat</span>
    </button>
  );
}
