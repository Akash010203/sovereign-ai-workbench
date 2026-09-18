import React, { useRef, useState, useEffect } from 'react';
import { ChevronDown, Check } from 'lucide-react';
import { useChatStore } from '../../store/chatStore';
import type { AgentMode } from '../../types';
import { cn } from '../../utils/cn';

const MODES: { value: AgentMode; label: string; description: string }[] = [
  { value: 'auto', label: 'Auto', description: 'Best model for the task' },
  { value: 'research', label: 'Research', description: 'In-depth knowledge retrieval' },
  { value: 'document', label: 'Document', description: 'PDF / DOCX analysis' },
  { value: 'vision', label: 'Vision', description: 'Image and diagram analysis' },
  { value: 'reasoning', label: 'Reasoning', description: 'Step-by-step reasoning' },
];

export default function ModeSelector() {
  const agentMode = useChatStore((s) => s.agentMode);
  const setAgentMode = useChatStore((s) => s.setAgentMode);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const current = MODES.find((m) => m.value === agentMode)!;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors py-1 px-2 rounded-md hover:bg-zinc-800/60"
      >
        <span>Mode: {current.label}</span>
        <ChevronDown size={11} className={cn('transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute bottom-full mb-2 left-0 w-52 bg-zinc-900 border border-zinc-700 rounded-xl shadow-xl z-50 py-1 overflow-hidden animate-fade-in">
          {MODES.map((m) => (
            <button
              key={m.value}
              onClick={() => { setAgentMode(m.value); setOpen(false); }}
              className="w-full flex items-start justify-between px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-800 transition-colors"
            >
              <div className="flex flex-col items-start">
                <span className="font-medium text-zinc-200">{m.label}</span>
                <span className="text-xs text-zinc-500">{m.description}</span>
              </div>
              {m.value === agentMode && <Check size={13} className="text-indigo-400 mt-0.5" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
