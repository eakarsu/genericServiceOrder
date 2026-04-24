import api from './client';
import type { DashboardStats, Order } from '../types';

export async function getStats(): Promise<DashboardStats> {
  const res = await api.get('/dashboard/stats');
  return res.data;
}

export async function getRecentOrders(): Promise<{ orders: Order[] }> {
  const res = await api.get('/dashboard/recent-orders');
  return res.data;
}
