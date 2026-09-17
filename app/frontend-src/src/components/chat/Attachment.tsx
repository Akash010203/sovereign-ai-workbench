import React from 'react';
import { X } from 'lucide-react';
import { getFileIcon, formatFileSize } from '../../utils/fileUtils';
import type { Attachment as AttachmentType } from '../../types';
import { cn } from '../../utils/cn';

interface Props {
  attachment: AttachmentType;
  onRemove?: (id: string) => void;
  readonly?: boolean;
}

export default function Attachment({ attachment, onRemove, readonly = false }: Props) {
  const icon = getFileIcon(attachment.type, attachment.name);

  const statusColor =
    attachment.status === 'error'
      ? 'border-red-500/40 bg-red-500/10'
      : attachment.status === 'uploading' || attachment.status === 'reading'
      ? 'border-indigo-500/40 bg-indigo-500/10 animate-pulse'
      : 'border-zinc-700 bg-zinc-800 dark:border-zinc-700 dark:bg-zinc-800/60';

  return (
    <div
      className={cn(
        'relative flex items-center gap-2 px-3 py-2 rounded-lg border text-sm',
        'max-w-[220px] group',
        statusColor,
      )}
    >
      {/* Upload progress bar */}
      {attachment.status === 'uploading' && (
        <div
          className="absolute bottom-0 left-0 h-0.5 bg-indigo-500 rounded-b-lg transition-all"
          style={{ width: `${attachment.progress}%` }}
        />
      )}

      <span className="text-base flex-shrink-0">{icon}</span>

      <div className="flex flex-col min-w-0">
        <span className="truncate text-zinc-200 dark:text-zinc-200 font-medium leading-tight">
          {attachment.name}
        </span>
        <span className="text-zinc-500 dark:text-zinc-500 text-xs">
          {attachment.status === 'uploading'
            ? `Uploading ${attachment.progress}%…`
            : attachment.status === 'reading'
            ? 'Reading…'
            : attachment.status === 'error'
            ? attachment.error ?? 'Error'
            : formatFileSize(attachment.size)}
        </span>
      </div>

      {!readonly && onRemove && (
        <button
          onClick={() => onRemove(attachment.id)}
          className="ml-auto flex-shrink-0 text-zinc-500 hover:text-zinc-200 transition-colors"
          aria-label="Remove attachment"
        >
          <X size={13} />
        </button>
      )}
    </div>
  );
}
