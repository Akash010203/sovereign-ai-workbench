import React, { useEffect, useState } from 'react';
import { useChatStore } from '../../store/chatStore';

/**
 * Renders a streaming placeholder message with a progressive text reveal.
 * When streaming ends (content arrives), it displays the full content.
 */
export default function StreamingMessage({ content }: { content: string }) {
  const streamingMessageId = useChatStore((s) => s.streamingMessageId);
  const [displayed, setDisplayed] = useState('');

  useEffect(() => {
    if (!content) return;
    // Reveal text progressively after it arrives
    let i = 0;
    setDisplayed('');
    const speed = Math.max(2, Math.floor(1000 / content.length)); // adaptive speed
    const interval = setInterval(() => {
      i += 3; // reveal 3 chars per tick
      setDisplayed(content.slice(0, i));
      if (i >= content.length) clearInterval(interval);
    }, speed);
    return () => clearInterval(interval);
  }, [content]);

  return <span>{displayed || content}</span>;
}
