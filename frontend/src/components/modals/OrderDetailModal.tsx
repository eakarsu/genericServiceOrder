import { useState } from 'react';
import { X, Edit2, Trash2, Save } from 'lucide-react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import type { Order } from '../../types';
import { orderEditSchema, type OrderEditForm } from '../../utils/validators';
import { ORDER_STATUSES } from '../../utils/constants';
import { formatDateTime, formatCurrency, formatSectorName, getStatusColor } from '../../utils/formatters';

interface OrderDetailModalProps {
  order: Order | null;
  onClose: () => void;
  onUpdate: (id: number, data: Partial<Order>) => Promise<void>;
  onDelete: (id: number) => void;
  canEdit: boolean;
}

export default function OrderDetailModal({ order, onClose, onUpdate, onDelete, canEdit }: OrderDetailModalProps) {
  const [editing, setEditing] = useState(false);
  const { register, handleSubmit, formState: { errors } } = useForm<OrderEditForm>({
    resolver: zodResolver(orderEditSchema),
    values: order ? {
      customer_name: order.customer_name,
      customer_phone: order.customer_phone || '',
      customer_email: order.customer_email || '',
      status: order.status,
      notes: order.notes || '',
    } : undefined,
  });

  if (!order) return null;

  const onSubmit = async (data: OrderEditForm) => {
    await onUpdate(order.id, data);
    setEditing(false);
  };

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold">Order #{order.id}</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600"><X size={20} /></button>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="p-4 space-y-4">
          {editing ? (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Customer Name</label>
                <input {...register('customer_name')} className="w-full border rounded-lg px-3 py-2 text-sm" />
                {errors.customer_name && <p className="text-xs text-red-500 mt-1">{errors.customer_name.message}</p>}
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                <input {...register('customer_phone')} className="w-full border rounded-lg px-3 py-2 text-sm" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input {...register('customer_email')} className="w-full border rounded-lg px-3 py-2 text-sm" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                <select {...register('status')} className="w-full border rounded-lg px-3 py-2 text-sm">
                  {ORDER_STATUSES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
                <textarea {...register('notes')} rows={3} className="w-full border rounded-lg px-3 py-2 text-sm" />
              </div>
            </>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div><span className="text-gray-500">Customer:</span><p className="font-medium">{order.customer_name}</p></div>
                <div><span className="text-gray-500">Phone:</span><p className="font-medium">{order.customer_phone || 'N/A'}</p></div>
                <div><span className="text-gray-500">Email:</span><p className="font-medium">{order.customer_email || 'N/A'}</p></div>
                <div><span className="text-gray-500">Sector:</span><p className="font-medium">{formatSectorName(order.sector)}</p></div>
                <div><span className="text-gray-500">Status:</span><p><span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(order.status)}`}>{order.status}</span></p></div>
                <div><span className="text-gray-500">Total:</span><p className="font-medium">{formatCurrency(order.total_amount)}</p></div>
                <div><span className="text-gray-500">Created:</span><p className="font-medium">{formatDateTime(order.created_at)}</p></div>
              </div>
              {order.items && order.items.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Items</h4>
                  <div className="bg-gray-50 rounded-lg p-3 space-y-1">
                    {order.items.map((item, i) => (
                      <div key={i} className="flex justify-between text-sm">
                        <span>{item.name} x{item.qty}</span>
                        <span className="text-gray-600">{formatCurrency(item.price * item.qty)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {order.notes && (
                <div><span className="text-sm text-gray-500">Notes:</span><p className="text-sm">{order.notes}</p></div>
              )}
            </>
          )}

          {canEdit && (
            <div className="flex justify-end gap-2 pt-2 border-t">
              {editing ? (
                <>
                  <button type="button" onClick={() => setEditing(false)} className="px-3 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200">Cancel</button>
                  <button type="submit" className="flex items-center gap-1 px-3 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700"><Save size={14} /> Save</button>
                </>
              ) : (
                <>
                  <button type="button" onClick={() => setEditing(true)} className="flex items-center gap-1 px-3 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700"><Edit2 size={14} /> Edit</button>
                  <button type="button" onClick={() => onDelete(order.id)} className="flex items-center gap-1 px-3 py-2 text-sm text-white bg-red-600 rounded-lg hover:bg-red-700"><Trash2 size={14} /> Delete</button>
                </>
              )}
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
