import type { Conversation, ConversationGroup } from '../types';

/**
 * Group conversations into Today / Yesterday / This Week / Older.
 */
export function groupConversations(convs: Conversation[]): ConversationGroup[] {
  const now = new Date();
  const today = startOfDay(now);
  const yesterday = startOfDay(new Date(now.getTime() - 86_400_000));
  const weekAgo = startOfDay(new Date(now.getTime() - 7 * 86_400_000));

  const groups: Record<string, Conversation[]> = {
    Today: [],
    Yesterday: [],
    'This week': [],
    Older: [],
  };

  for (const conv of convs) {
    const d = new Date(conv.updated_at || conv.created_at);
    if (d >= today) {
      groups['Today'].push(conv);
    } else if (d >= yesterday) {
      groups['Yesterday'].push(conv);
    } else if (d >= weekAgo) {
      groups['This week'].push(conv);
    } else {
      groups['Older'].push(conv);
    }
  }

  return Object.entries(groups)
    .filter(([, list]) => list.length > 0)
    .map(([label, conversations]) => ({ label, conversations }));
}

function startOfDay(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}

/** Format a timestamp string into a human-readable relative label */
export function formatRelativeDate(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    const mins = Math.floor(diff / 60_000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
}

/** Truncate a string to max length with ellipsis */
export function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max).trimEnd() + '…';
}
