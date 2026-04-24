import { useEffect, useState } from 'react';
import { ShoppingCart, DollarSign, TrendingUp, Building } from 'lucide-react';
import { getStats, getRecentOrders } from '../api/dashboard';
import { getSectors } from '../api/sectors';
import type { DashboardStats, Order, Sector } from '../types';
import StatsCard from '../components/ui/StatsCard';
import SectorCard from '../components/ui/SectorCard';
import { SkeletonCard } from '../components/ui/SkeletonLoader';
import { formatCurrency, formatRelative, getStatusColor, formatSectorName } from '../utils/formatters';

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recent, setRecent] = useState<Order[]>([]);
  const [sectors, setSectors] = useState<Sector[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getStats(), getRecentOrders(), getSectors()])
      .then(([s, r, sec]) => {
        setStats(s);
        setRecent(r.orders);
        setSectors(sec.sectors);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => <SkeletonCard key={i} />)}
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>

      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatsCard title="Total Orders" value={stats.total_orders} icon={ShoppingCart} color="bg-blue-500" />
          <StatsCard title="Total Revenue" value={formatCurrency(stats.total_revenue)} icon={DollarSign} color="bg-green-500" />
          <StatsCard title="Orders Today" value={stats.orders_today} icon={TrendingUp} color="bg-purple-500" />
          <StatsCard title="Active Sectors" value={stats.active_sectors} icon={Building} color="bg-orange-500" />
        </div>
      )}

      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Sectors</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {sectors.map(sector => (
            <SectorCard
              key={sector.id}
              sector={sector}
              orderCount={stats?.orders_by_sector[sector.name] || 0}
            />
          ))}
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Recent Orders</h2>
        <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-600">
              <tr>
                <th className="px-4 py-3 text-left font-medium">ID</th>
                <th className="px-4 py-3 text-left font-medium">Customer</th>
                <th className="px-4 py-3 text-left font-medium hidden md:table-cell">Sector</th>
                <th className="px-4 py-3 text-left font-medium">Status</th>
                <th className="px-4 py-3 text-right font-medium">Amount</th>
                <th className="px-4 py-3 text-right font-medium hidden sm:table-cell">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {recent.map(order => (
                <tr key={order.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-gray-500">#{order.id}</td>
                  <td className="px-4 py-3 font-medium text-gray-900">{order.customer_name}</td>
                  <td className="px-4 py-3 hidden md:table-cell text-gray-600">{formatSectorName(order.sector)}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(order.status)}`}>
                      {order.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-medium">{formatCurrency(order.total_amount)}</td>
                  <td className="px-4 py-3 text-right text-gray-500 hidden sm:table-cell">{formatRelative(order.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
