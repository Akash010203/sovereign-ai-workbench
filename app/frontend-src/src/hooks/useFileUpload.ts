import { useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { documentsApi } from '../api/documents';
import { generateId, isTextReadable, readFileAsText } from '../utils/fileUtils';
import type { Attachment } from '../types';

export function useFileUpload() {
  const store = useChatStore();

  /**
   * Add files to the pending attachments list.
   * Text-readable files are read immediately; binary files are queued.
   */
  const addFiles = useCallback(
    async (files: FileList | File[]) => {
      const fileArray = Array.from(files);

      for (const file of fileArray) {
        const id = generateId();
        const attachment: Attachment = {
          id,
          name: file.name,
          size: file.size,
          type: file.type,
          file,
          status: 'pending',
          progress: 0,
        };
        store.addAttachment(attachment);

        // Read text files immediately so they're ready to ingest
        if (isTextReadable(file)) {
          store.updateAttachment(id, { status: 'reading' });
          try {
            const text = await readFileAsText(file);
            store.updateAttachment(id, { status: 'done', textContent: text, progress: 100 });
          } catch {
            store.updateAttachment(id, { status: 'error', error: 'Could not read file' });
          }
        } else {
          // Binary files (PDF, DOCX, etc.) — marked pending, filename used as context
          store.updateAttachment(id, { status: 'done', progress: 100 });
        }
      }
    },
    [store],
  );

  /**
   * Upload all "done" text attachments to the RAG knowledge base.
   * Returns attachment metadata to embed in the chat message.
   */
  const uploadAndIngest = useCallback(async (): Promise<{
    metas: Array<{ id: string; name: string; size: number; type: string }>;
  }> => {
    const metas: Array<{ id: string; name: string; size: number; type: string }> = [];

    for (const att of store.attachments) {
      metas.push({ id: att.id, name: att.name, size: att.size, type: att.type });

      if (att.textContent && att.status === 'done') {
        // Ingest text into RAG
        store.updateAttachment(att.id, { status: 'uploading', progress: 50 });
        try {
          await documentsApi.ingestText(att.textContent, att.name);
          store.updateAttachment(att.id, { status: 'done', progress: 100 });
        } catch (err) {
          const msg = err instanceof Error ? err.message : 'Upload failed';
          store.updateAttachment(att.id, { status: 'error', error: msg });
        }
      }
    }

    return { metas };
  }, [store]);

  const removeAttachment = useCallback(
    (id: string) => store.removeAttachment(id),
    [store],
  );

  return { addFiles, uploadAndIngest, removeAttachment, attachments: store.attachments };
}
