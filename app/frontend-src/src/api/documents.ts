import { apiClient } from './client';

export interface IngestResult {
  chunks_added: number;
  total_chunks: number;
}

export interface SearchResult {
  source: string;
  text: string;
  score: number;
  citation: string;
}

export const documentsApi = {
  /**
   * Ingest plain text into the local RAG knowledge base.
   * For text/csv/md files: read client-side, send here.
   * For binary files (PDF, DOCX, etc.): include filename as source context.
   */
  async ingestText(text: string, source: string): Promise<IngestResult> {
    const { data } = await apiClient.post<IngestResult>('/api/rag/ingest', {
      text,
      source,
    });
    return data;
  },

  /** Semantic search over the knowledge base */
  async search(
    query: string,
    top_k = 5,
  ): Promise<{ results: SearchResult[]; citations: string }> {
    const { data } = await apiClient.post('/api/rag/search', { query, top_k });
    return data;
  },
};
