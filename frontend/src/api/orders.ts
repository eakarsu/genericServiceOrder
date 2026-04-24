import api from './client';
import type { Order, PaginatedOrders, OrderFilters } from '../types';

export async function getOrders(filters: OrderFilters = {}): Promise<PaginatedOrders> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.append(key, String(value));
  });
  const res = await api.get(`/orders?${params.toString()}`);
  return res.data;
}

export async function getOrder(id: number): Promise<Order> {
  const res = await api.get(`/orders/${id}`);
  return res.data;
}

export async function updateOrder(id: number, data: Partial<Order>): Promise<Order> {
  const res = await api.put(`/orders/${id}`, data);
  return res.data;
}

export async function deleteOrder(id: number) {
  const res = await api.delete(`/orders/${id}`);
  return res.data;
}

export async function bulkDeleteOrders(ids: number[]) {
  const res = await api.post('/orders/bulk-delete', { ids });
  return res.data;
}

export async function bulkUpdateOrders(ids: number[], status: string) {
  const res = await api.put('/orders/bulk-update', { ids, status });
  return res.data;
}
