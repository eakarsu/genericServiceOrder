import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { getSector } from '../api/sectors';
import type { Sector, Order } from '../types';
import { SECTOR_ICONS } from '../utils/constants';
import { formatCurrency, formatDate, getStatusColor, formatSectorName } from '../utils/formatters';
import { SkeletonCard } from '../components/ui/SkeletonLoader';
import EmptyState from '../components/ui/EmptyState';

export default function SectorDetailPage() {
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const [sector, setSector] = useState<Sector | null>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [orderCount, setOrderCount] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!name) return;
    getSector(name)
      .then(data => {
        setSector(data.sector);
        setOrders(data.recent_orders || []);
        setOrderCount(data.order_count || 0);
      })
      .catch(() => navigate('/'))
      .finally(() => setLoading(false));
  }, [name, navigate]);

  if (loading) {
    return (
      <div className="space-y-4">
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  if (!sector) return null;

  const emoji = SECTOR_ICONS[sector.icon || ''] || '📦';

  return (
    <div className="space-y-6">
      <button onClick={() => navigate('/')} className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900">
        <ArrowLeft size={16} /> Back to Dashboard
      </button>

      <div className="bg-white rounded-xl p-6 shadow-sm border">
        <div className="flex items-center gap-4">
          <span className="text-4xl">{emoji}</span>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{sector.display_name}</h1>
            <p className="text-gray-500">{sector.description}</p>
            <p className="text-sm text-gray-400 mt-1">{orderCount} total orders</p>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Recent Orders</h2>
        {orders.length === 0 ? (
          <EmptyState title="No orders yet" message={`No orders for ${formatSectorName(sector.name)} yet.`} />
        ) : (
          <div className="bg-white rounded-xl shadow-sm border overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-600">
                <tr>
                  <th className="px-4 py-3 text-left font-medium">ID</th>
                  <th className="px-4 py-3 text-left font-medium">Customer</th>
                  <th className="px-4 py-3 text-left font-medium">Status</th>
                  <th className="px-4 py-3 text-right font-medium">Amount</th>
                  <th className="px-4 py-3 text-right font-medium">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {orders.map(o => (
                  <tr key={o.id} className="hover:bg-gray-50 cursor-pointer" onClick={() => navigate('/orders')}>
                    <td className="px-4 py-3 text-gray-500">#{o.id}</td>
                    <td className="px-4 py-3 font-medium text-gray-900">{o.customer_name}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(o.status)}`}>{o.status}</span>
                    </td>
                    <td className="px-4 py-3 text-right font-medium">{formatCurrency(o.total_amount)}</td>
                    <td className="px-4 py-3 text-right text-gray-500">{formatDate(o.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
