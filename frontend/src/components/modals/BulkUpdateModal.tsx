import { useState } from 'react';
import { X } from 'lucide-react';
import { ORDER_STATUSES } from '../../utils/constants';

interface BulkUpdateModalProps {
  open: boolean;
  count: number;
  onConfirm: (status: string) => void;
  onCancel: () => void;
}

export default function BulkUpdateModal({ open, count, onConfirm, onCancel }: BulkUpdateModalProps) {
  const [status, setStatus] = useState('');

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/50" onClick={onCancel}>
      <div className="bg-white rounded-xl shadow-xl p-6 max-w-sm w-full mx-4" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Update {count} Orders</h3>
          <button onClick={onCancel} className="text-gray-400 hover:text-gray-600"><X size={20} /></button>
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">New Status</label>
          <select
            value={status}
            onChange={e => setStatus(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">Select status...</option>
            {ORDER_STATUSES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
          </select>
        </div>
        <div className="flex justify-end gap-2">
          <button onClick={onCancel} className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200">Cancel</button>
          <button
            onClick={() => status && onConfirm(status)}
            disabled={!status}
            className="px-4 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            Update
          </button>
        </div>
      </div>
    </div>
  );
}
