import api from './client';
import type { User, PaginatedUsers } from '../types';

export async function getUsers(params: { page?: number; page_size?: number; search?: string; role_id?: number } = {}): Promise<PaginatedUsers> {
  const res = await api.get('/users', { params });
  return res.data;
}

export async function getUser(id: number): Promise<User> {
  const res = await api.get(`/users/${id}`);
  return res.data;
}

export async function createUser(data: { email: string; password: string; name: string; phone?: string; role_id: number }) {
  const res = await api.post('/users', data);
  return res.data;
}

export async function updateUser(id: number, data: Partial<User>) {
  const res = await api.put(`/users/${id}`, data);
  return res.data;
}

export async function deleteUser(id: number) {
  const res = await api.delete(`/users/${id}`);
  return res.data;
}

export async function getProfile(): Promise<User> {
  const res = await api.get('/users/me/profile');
  return res.data;
}

export async function updateProfile(data: { name?: string; phone?: string }) {
  const res = await api.put('/users/me/profile', data);
  return res.data;
}
