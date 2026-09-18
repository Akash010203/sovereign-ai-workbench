import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Copy, RefreshCw, ThumbsUp, ThumbsDown, Check } from 'lucide-react';
import type { Message as MessageType } from '../../types';
import TypingIndicator from './TypingIndicator';
import Attachment from './Attachment';
import { cn } from '../../utils/cn';

interface Props {
  message: MessageType;
  isStreaming?: boolean;
  onRegenerate?: () => void;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button onClick={handleCopy} className="action-btn" title="Copy">
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

function CodeBlock({ className, children, ...props }: React.ComponentPropsWithoutRef<'code'>) {
  const isBlock = !props.style;
  const [copied, setCopied] = useState(false);
  const code = String(children).replace(/\n$/, '');

  if (!isBlock && !className) {
    return (
      <code className="inline-code" {...props}>
        {children}
      </code>
    );
  }

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group my-4">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-xs px-2 py-1 rounded bg-zinc-700 text-zinc-300 hover:bg-zinc-600 flex items-center gap-1"
      >
        {copied ? <Check size={11} /> : <Copy size={11} />}
        {copied ? 'Copied' : 'Copy'}
      </button>
      <code className={className} {...props}>
        {children}
      </code>
    </div>
  );
}

export default function Message({ message, isStreaming, onRegenerate }: Props) {
  const isUser = message.role === 'user';
  const isEmpty = !message.content && isStreaming;

  return (
    <div className={cn('message-row', isUser ? 'message-row--user' : 'message-row--ai')}>
      {/* Avatar */}
      {!isUser && (
        <div className="ai-avatar">
          <span className="text-xs font-bold text-indigo-400">AI</span>
        </div>
      )}

      <div className={cn('message-content', isUser ? 'message-content--user' : 'message-content--ai')}>
        {/* Attachments (shown above message for user) */}
        {isUser && message.attachments && message.attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {message.attachments.map((a) => (
              <Attachment
                key={a.id}
                attachment={{ ...a, file: null as any, status: 'done', progress: 100 }}
                readonly
              />
            ))}
          </div>
        )}

        {/* Message body */}
        {isEmpty ? (
          <TypingIndicator />
        ) : isUser ? (
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                code: CodeBlock as any,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* RAG sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-zinc-800">
            <p className="text-xs text-zinc-500 mb-1.5">Sources</p>
            <div className="flex flex-wrap gap-1.5">
              {message.sources.map((src, i) => (
                <span
                  key={i}
                  className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 border border-zinc-700"
                >
                  {src.source}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* AI message actions */}
        {!isUser && !isEmpty && message.content && (
          <div className="flex items-center gap-1 mt-2">
            <CopyButton text={message.content} />
            {onRegenerate && (
              <button onClick={onRegenerate} className="action-btn" title="Regenerate">
                <RefreshCw size={13} />
              </button>
            )}
            <button className="action-btn" title="Good response">
              <ThumbsUp size={13} />
            </button>
            <button className="action-btn" title="Bad response">
              <ThumbsDown size={13} />
            </button>
            {message.model_used && (
              <span className="ml-auto text-xs text-zinc-600">{message.model_used}</span>
            )}
          </div>
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="user-avatar">
          <span className="text-xs font-bold text-zinc-300">You</span>
        </div>
      )}
    </div>
  );
}
