import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, Check } from 'lucide-react';
import { useChatStore } from '../../store/chatStore';
import { chatApi } from '../../api/chat';
import { cn } from '../../utils/cn';

export default function ModelSelector() {
  const models = useChatStore((s) => s.models);
  const selectedModel = useChatStore((s) => s.selectedModel);
  const setModels = useChatStore((s) => s.setModels);
  const setSelectedModel = useChatStore((s) => s.setSelectedModel);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Load models on mount
  useEffect(() => {
    chatApi.getModels().then((ms) => {
      setModels(ms);
      if (!selectedModel && ms.length > 0) {
        const available = ms.find((m) => m.available) ?? ms[0];
        setSelectedModel(available.name);
      }
    }).catch(() => {});
  }, []);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const currentModel = models.find((m) => m.name === selectedModel);
  const displayName = currentModel?.name ?? 'Sovereign AI';

  if (models.length === 0) {
    return (
      <span className="text-sm text-zinc-400 px-3 py-1">Sovereign AI</span>
    );
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-sm text-zinc-300 hover:text-zinc-100 transition-colors px-2 py-1 rounded-md hover:bg-zinc-800"
      >
        <span className="font-medium">{displayName}</span>
        <ChevronDown size={13} className={cn('transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute top-full mt-1 left-1/2 -translate-x-1/2 w-52 bg-zinc-900 border border-zinc-700 rounded-xl shadow-xl z-50 py-1 overflow-hidden animate-fade-in">
          {models.map((m) => (
            <button
              key={m.name}
              onClick={() => { setSelectedModel(m.name); setOpen(false); }}
              disabled={!m.available}
              className={cn(
                'w-full flex items-center justify-between px-3 py-2 text-sm transition-colors',
                m.available
                  ? 'text-zinc-200 hover:bg-zinc-800 cursor-pointer'
                  : 'text-zinc-600 cursor-not-allowed',
              )}
            >
              <div className="flex flex-col items-start">
                <span className="font-medium">{m.name}</span>
                <span className="text-xs text-zinc-500">{m.model_type}</span>
              </div>
              <div className="flex items-center gap-2">
                {!m.available && (
                  <span className="text-xs text-zinc-600 bg-zinc-800 px-1.5 py-0.5 rounded">
                    offline
                  </span>
                )}
                {m.name === selectedModel && <Check size={13} className="text-indigo-400" />}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
