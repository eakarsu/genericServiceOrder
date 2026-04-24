import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  message?: string;
}

export default function EmptyState({ title = 'No data found', message = 'Try adjusting your filters or search terms.' }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-gray-400">
      <Inbox size={48} className="mb-3" />
      <p className="text-lg font-medium text-gray-500">{title}</p>
      <p className="text-sm">{message}</p>
    </div>
  );
}
