import api from './client';
import type { TokenResponse, User } from '../types';

export async function login(email: string, password: string): Promise<TokenResponse> {
  const res = await api.post('/auth/login', { email, password });
  return res.data;
}

export async function register(data: { email: string; password: string; name: string; phone?: string }) {
  const res = await api.post('/auth/register', data);
  return res.data;
}

export async function logout() {
  const res = await api.post('/auth/logout');
  return res.data;
}

export async function refreshToken(refresh_token: string): Promise<TokenResponse> {
  const res = await api.post('/auth/refresh', { refresh_token });
  return res.data;
}

export async function forgotPassword(email: string) {
  const res = await api.post('/auth/forgot-password', { email });
  return res.data;
}

export async function resetPassword(token: string, new_password: string) {
  const res = await api.post('/auth/reset-password', { token, new_password });
  return res.data;
}

export async function changePassword(current_password: string, new_password: string) {
  const res = await api.post('/auth/change-password', { current_password, new_password });
  return res.data;
}

export async function verifyEmail(token: string) {
  const res = await api.get(`/auth/verify-email/${token}`);
  return res.data;
}

export async function getMe(): Promise<User> {
  const res = await api.get('/auth/me');
  return res.data;
}
