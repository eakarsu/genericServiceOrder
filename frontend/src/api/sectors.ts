import api from './client';
import type { Sector } from '../types';

export async function getSectors(): Promise<{ sectors: Sector[]; total: number }> {
  const res = await api.get('/sectors');
  return res.data;
}

export async function getSector(name: string) {
  const res = await api.get(`/sectors/${name}`);
  return res.data;
}
