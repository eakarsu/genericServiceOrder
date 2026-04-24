import { Trash2, RefreshCw } from 'lucide-react';

interface BulkActionBarProps {
  selectedCount: number;
  onBulkDelete: () => void;
  onBulkUpdate: () => void;
  onClearSelection: () => void;
}

export default function BulkActionBar({ selectedCount, onBulkDelete, onBulkUpdate, onClearSelection }: BulkActionBarProps) {
  if (selectedCount === 0) return null;

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 flex items-center justify-between">
      <span className="text-sm text-blue-800 font-medium">{selectedCount} item(s) selected</span>
      <div className="flex items-center gap-2">
        <button
          onClick={onBulkUpdate}
          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <RefreshCw size={14} /> Update Status
        </button>
        <button
          onClick={onBulkDelete}
          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700"
        >
          <Trash2 size={14} /> Delete
        </button>
        <button onClick={onClearSelection} className="text-sm text-gray-600 hover:text-gray-800 ml-2">
          Clear
        </button>
      </div>
    </div>
  );
}
