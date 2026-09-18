import React, { useRef, useEffect, KeyboardEvent, useCallback } from 'react';
import { Send, Paperclip, Square } from 'lucide-react';
import { useChatStore } from '../../store/chatStore';
import { useChat } from '../../hooks/useChat';
import { useFileUpload } from '../../hooks/useFileUpload';
import Attachment from './Attachment';
import ModeSelector from '../common/ModeSelector';
import { ACCEPTED_FILE_TYPES } from '../../utils/fileUtils';
import { cn } from '../../utils/cn';

export default function Composer() {
  const textRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isLoading = useChatStore((s) => s.isLoading);
  const attachments = useChatStore((s) => s.attachments);

  const { sendMessage } = useChat();
  const { addFiles, uploadAndIngest, removeAttachment } = useFileUpload();

  // Auto-grow textarea
  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  });

  const handleSend = useCallback(async () => {
    const text = textRef.current?.value.trim() ?? '';
    if (!text || isLoading) return;
    if (textRef.current) textRef.current.value = '';
    if (textRef.current) textRef.current.style.height = 'auto';

    // Upload/ingest any text attachments first
    const { metas } = await uploadAndIngest();
    await sendMessage(text, metas);
  }, [isLoading, uploadAndIngest, sendMessage]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(e.target.files);
      e.target.value = '';
    }
  };

  return (
    <div className="composer-wrapper">
      {/* Attachment chips above composer */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 px-4 pb-2">
          {attachments.map((a) => (
            <Attachment key={a.id} attachment={a} onRemove={removeAttachment} />
          ))}
        </div>
      )}

      {/* Composer box */}
      <div className="composer-box">
        {/* Textarea */}
        <textarea
          ref={textRef}
          className="composer-textarea"
          placeholder="Ask anything…"
          rows={1}
          disabled={isLoading}
          onKeyDown={handleKeyDown}
        />

        {/* Bottom row: attach | mode | send */}
        <div className="flex items-center justify-between mt-2 px-1">
          <div className="flex items-center gap-2">
            {/* File picker */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ACCEPTED_FILE_TYPES}
              className="hidden"
              onChange={handleFileChange}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                'composer-icon-btn',
                attachments.length > 0 && 'text-indigo-400',
              )}
              title="Attach files"
              disabled={isLoading}
            >
              <Paperclip size={16} />
              {attachments.length > 0 && (
                <span className="ml-1 text-xs font-medium">{attachments.length}</span>
              )}
            </button>

            {/* Agent mode selector */}
            <ModeSelector />
          </div>

          {/* Send / Stop button */}
          {isLoading ? (
            <button className="send-btn send-btn--stop" title="Stop generating">
              <Square size={14} fill="currentColor" />
              <span className="text-xs">Stop</span>
            </button>
          ) : (
            <button
              onClick={handleSend}
              className="send-btn"
              title="Send message"
            >
              <Send size={14} />
            </button>
          )}
        </div>
      </div>

      <p className="text-center text-xs text-zinc-600 mt-2">
        SovereignAI · All data stays on your device · SIH 2026
      </p>
    </div>
  );
}
