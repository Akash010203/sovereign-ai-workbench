import React, { useCallback, useEffect, useState } from 'react';
import { Upload } from 'lucide-react';

interface Props {
  onDrop: (files: File[]) => void;
}

export default function DragDropOverlay({ onDrop }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [dragCounter, setDragCounter] = useState(0);

  const handleDragEnter = useCallback((e: DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer?.types.includes('Files')) {
      setDragCounter((c) => c + 1);
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragCounter((c) => {
      const next = c - 1;
      if (next <= 0) setIsDragging(false);
      return next;
    });
  }, []);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      setDragCounter(0);
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length > 0) onDrop(files);
    },
    [onDrop],
  );

  useEffect(() => {
    document.addEventListener('dragenter', handleDragEnter);
    document.addEventListener('dragleave', handleDragLeave);
    document.addEventListener('dragover', handleDragOver);
    document.addEventListener('drop', handleDrop);
    return () => {
      document.removeEventListener('dragenter', handleDragEnter);
      document.removeEventListener('dragleave', handleDragLeave);
      document.removeEventListener('dragover', handleDragOver);
      document.removeEventListener('drop', handleDrop);
    };
  }, [handleDragEnter, handleDragLeave, handleDragOver, handleDrop]);

  if (!isDragging) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm pointer-events-none">
      <div className="flex flex-col items-center gap-4 p-12 rounded-2xl border-2 border-dashed border-indigo-500/70 bg-indigo-500/10">
        <Upload size={40} className="text-indigo-400" />
        <p className="text-xl font-medium text-indigo-200">Drop files to upload</p>
        <p className="text-sm text-indigo-300/60">PDF, DOCX, XLSX, PPTX, TXT, CSV, images…</p>
      </div>
    </div>
  );
}
