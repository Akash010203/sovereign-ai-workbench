/** Format bytes to human-readable string: 1024 → "1 KB" */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Map MIME type to an emoji icon for display */
export function getFileIcon(type: string, name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() ?? '';
  if (type.startsWith('image/')) return '🖼️';
  if (ext === 'pdf' || type === 'application/pdf') return '📄';
  if (ext === 'docx' || ext === 'doc') return '📝';
  if (ext === 'xlsx' || ext === 'xls' || ext === 'csv') return '📊';
  if (ext === 'pptx' || ext === 'ppt') return '📑';
  if (ext === 'txt' || ext === 'md') return '📃';
  return '📎';
}

/** Determine if a file can be read as plain text client-side */
export function isTextReadable(file: File): boolean {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  const textExts = new Set(['txt', 'csv', 'md', 'json', 'xml', 'log', 'yaml', 'yml', 'toml']);
  return textExts.has(ext) || file.type.startsWith('text/');
}

/** Read a File as UTF-8 text */
export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file, 'utf-8');
  });
}

/** Accepted file extensions for the composer's file picker */
export const ACCEPTED_FILE_TYPES =
  '.pdf,.docx,.doc,.xlsx,.xls,.pptx,.ppt,.txt,.csv,.md,.png,.jpg,.jpeg,.json,.yaml,.yml';

/** Generate a stable random ID */
export function generateId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
