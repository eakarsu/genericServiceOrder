import { useState, useEffect, useCallback } from 'react';
import { ArrowUpDown, Download } from 'lucide-react';
import { getOrders, updateOrder, deleteOrder, bulkDeleteOrders, bulkUpdateOrders } from '../api/orders';
import { getSectors } from '../api/sectors';
import { exportCSV, exportPDF, downloadBlob } from '../api/exportApi';
import type { Order, Sector, OrderFilters } from '../types';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import { useDebounce } from '../hooks/useDebounce';
import SearchBar from '../components/ui/SearchBar';
import FilterControls from '../components/ui/FilterControls';
import Pagination from '../components/ui/Pagination';
import EmptyState from '../components/ui/EmptyState';
import BulkActionBar from '../components/ui/BulkActionBar';
import ConfirmDialog from '../components/ui/ConfirmDialog';
import OrderDetailModal from '../components/modals/OrderDetailModal';
import BulkUpdateModal from '../components/modals/BulkUpdateModal';
import { SkeletonTable } from '../components/ui/SkeletonLoader';
import { formatCurrency, formatDate, formatSectorName, getStatusColor } from '../utils/formatters';

export default function OrdersPage() {
  const { user } = useAuth();
  const { addToast } = useToast();
  const canEdit = user?.role_name === 'admin' || user?.role_name === 'manager';

  const [orders, setOrders] = useState<Order[]>([]);
  const [sectors, setSectors] = useState<Sector[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);

  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('');
  const [sector, setSector] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [page, setPage] = useState(1);

  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [selectedOrder, setSelectedOrder] = useState<Order | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [showBulkDelete, setShowBulkDelete] = useState(false);
  const [showBulkUpdate, setShowBulkUpdate] = useState(false);

  const debouncedSearch = useDebounce(search);

  const fetchOrders = useCallback(async () => {
    setLoading(true);
    try {
      const filters: OrderFilters = {
        page, page_size: 20, sort_by: sortBy, sort_order: sortOrder,
        ...(debouncedSearch && { search: debouncedSearch }),
        ...(status && { status }),
        ...(sector && { sector }),
        ...(dateFrom && { date_from: dateFrom }),
        ...(dateTo && { date_to: dateTo }),
      };
      const data = await getOrders(filters);
      setOrders(data.orders);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch {
      addToast('Failed to load orders', 'error');
    } finally {
      setLoading(false);
    }
  }, [page, sortBy, sortOrder, debouncedSearch, status, sector, dateFrom, dateTo, addToast]);

  useEffect(() => { fetchOrders(); }, [fetchOrders]);
  useEffect(() => {
    getSectors().then(d => setSectors(d.sectors)).catch(() => {});
  }, []);

  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
    setPage(1);
  };

  const toggleSelect = (id: number) => {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === orders.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(orders.map(o => o.id)));
    }
  };

  const handleUpdate = async (id: number, data: Partial<Order>) => {
    try {
      await updateOrder(id, data);
      addToast('Order updated', 'success');
      fetchOrders();
    } catch {
      addToast('Failed to update order', 'error');
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteOrder(id);
      addToast('Order deleted', 'success');
      setSelectedOrder(null);
      setConfirmDelete(null);
      fetchOrders();
    } catch {
      addToast('Failed to delete order', 'error');
    }
  };

  const handleBulkDelete = async () => {
    try {
      await bulkDeleteOrders(Array.from(selected));
      addToast(`Deleted ${selected.size} orders`, 'success');
      setSelected(new Set());
      setShowBulkDelete(false);
      fetchOrders();
    } catch {
      addToast('Bulk delete failed', 'error');
    }
  };

  const handleBulkUpdate = async (newStatus: string) => {
    try {
      await bulkUpdateOrders(Array.from(selected), newStatus);
      addToast(`Updated ${selected.size} orders`, 'success');
      setSelected(new Set());
      setShowBulkUpdate(false);
      fetchOrders();
    } catch {
      addToast('Bulk update failed', 'error');
    }
  };

  const handleExportCSV = async () => {
    try {
      const blob = await exportCSV({ sector, status, date_from: dateFrom, date_to: dateTo, search: debouncedSearch });
      downloadBlob(blob, 'orders.csv');
      addToast('CSV exported', 'success');
    } catch {
      addToast('Export failed', 'error');
    }
  };

  const handleExportPDF = async () => {
    try {
      const blob = await exportPDF({ sector, status, date_from: dateFrom, date_to: dateTo, search: debouncedSearch });
      downloadBlob(blob, 'orders.pdf');
      addToast('PDF exported', 'success');
    } catch {
      addToast('Export failed', 'error');
    }
  };

  const clearFilters = () => {
    setStatus(''); setSector(''); setDateFrom(''); setDateTo(''); setPage(1);
  };

  const SortHeader = ({ field, children }: { field: string; children: React.ReactNode }) => (
    <th
      className="px-4 py-3 text-left font-medium cursor-pointer hover:text-blue-600 select-none"
      onClick={() => handleSort(field)}
    >
      <span className="flex items-center gap-1">
        {children}
        <ArrowUpDown size={12} className={sortBy === field ? 'text-blue-600' : 'text-gray-400'} />
      </span>
    </th>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Orders</h1>
        <div className="flex items-center gap-2">
          <button onClick={handleExportCSV} className="flex items-center gap-1 px-3 py-2 text-sm border rounded-lg hover:bg-gray-50">
            <Download size={14} /> CSV
          </button>
          <button onClick={handleExportPDF} className="flex items-center gap-1 px-3 py-2 text-sm border rounded-lg hover:bg-gray-50">
            <Download size={14} /> PDF
          </button>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <div className="w-full sm:w-72">
          <SearchBar value={search} onChange={v => { setSearch(v); setPage(1); }} placeholder="Search orders..." />
        </div>
        <FilterControls
          status={status} sector={sector} dateFrom={dateFrom} dateTo={dateTo} sectors={sectors}
          onStatusChange={v => { setStatus(v); setPage(1); }}
          onSectorChange={v => { setSector(v); setPage(1); }}
          onDateFromChange={v => { setDateFrom(v); setPage(1); }}
          onDateToChange={v => { setDateTo(v); setPage(1); }}
          onClear={clearFilters}
        />
      </div>

      {canEdit && <BulkActionBar selectedCount={selected.size} onBulkDelete={() => setShowBulkDelete(true)} onBulkUpdate={() => setShowBulkUpdate(true)} onClearSelection={() => setSelected(new Set())} />}

      <div className="bg-white rounded-xl shadow-sm border overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              {canEdit && (
                <th className="px-4 py-3 w-8">
                  <input type="checkbox" checked={orders.length > 0 && selected.size === orders.length} onChange={toggleAll} className="rounded" />
                </th>
              )}
              <SortHeader field="id">ID</SortHeader>
              <SortHeader field="customer_name">Customer</SortHeader>
              <th className="px-4 py-3 text-left font-medium hidden lg:table-cell">Sector</th>
              <SortHeader field="status">Status</SortHeader>
              <SortHeader field="total_amount">Amount</SortHeader>
              <SortHeader field="created_at">Date</SortHeader>
            </tr>
          </thead>
          {loading ? (
            <SkeletonTable rows={10} cols={canEdit ? 7 : 6} />
          ) : orders.length === 0 ? (
            <tbody><tr><td colSpan={7}><EmptyState title="No orders found" /></td></tr></tbody>
          ) : (
            <tbody className="divide-y divide-gray-100">
              {orders.map(order => (
                <tr key={order.id} className="hover:bg-gray-50 cursor-pointer" onClick={() => setSelectedOrder(order)}>
                  {canEdit && (
                    <td className="px-4 py-3" onClick={e => e.stopPropagation()}>
                      <input type="checkbox" checked={selected.has(order.id)} onChange={() => toggleSelect(order.id)} className="rounded" />
                    </td>
                  )}
                  <td className="px-4 py-3 text-gray-500">#{order.id}</td>
                  <td className="px-4 py-3 font-medium text-gray-900">{order.customer_name}</td>
                  <td className="px-4 py-3 hidden lg:table-cell text-gray-600">{formatSectorName(order.sector)}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(order.status)}`}>{order.status}</span>
                  </td>
                  <td className="px-4 py-3 font-medium">{formatCurrency(order.total_amount)}</td>
                  <td className="px-4 py-3 text-gray-500">{formatDate(order.created_at)}</td>
                </tr>
              ))}
            </tbody>
          )}
        </table>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-500">{total} orders total</p>
        <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
      </div>

      <OrderDetailModal
        order={selectedOrder}
        onClose={() => setSelectedOrder(null)}
        onUpdate={handleUpdate}
        onDelete={id => { setSelectedOrder(null); setConfirmDelete(id); }}
        canEdit={canEdit}
      />

      <ConfirmDialog
        open={confirmDelete !== null}
        title="Delete Order"
        message="Are you sure you want to delete this order? This action cannot be undone."
        confirmLabel="Delete"
        onConfirm={() => confirmDelete && handleDelete(confirmDelete)}
        onCancel={() => setConfirmDelete(null)}
      />

      <ConfirmDialog
        open={showBulkDelete}
        title="Delete Selected Orders"
        message={`Are you sure you want to delete ${selected.size} orders?`}
        confirmLabel="Delete All"
        onConfirm={handleBulkDelete}
        onCancel={() => setShowBulkDelete(false)}
      />

      <BulkUpdateModal
        open={showBulkUpdate}
        count={selected.size}
        onConfirm={handleBulkUpdate}
        onCancel={() => setShowBulkUpdate(false)}
      />
    </div>
  );
}
