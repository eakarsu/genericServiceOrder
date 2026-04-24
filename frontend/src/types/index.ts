export interface User {
  id: number;
  email: string;
  name: string;
  phone: string | null;
  role_id: number;
  role_name: string | null;
  is_active: boolean;
  is_email_verified: boolean;
  created_at: string;
  updated_at: string | null;
}

export interface Role {
  id: number;
  name: string;
  display_name: string;
  permissions: Record<string, string[]>;
}

export interface Order {
  id: number;
  customer_name: string;
  customer_phone: string | null;
  customer_email: string | null;
  sector: string;
  status: string;
  items: OrderItem[] | null;
  total_amount: number;
  notes: string | null;
  created_at: string;
  updated_at: string | null;
}

export interface OrderItem {
  name: string;
  qty: number;
  price: number;
}

export interface Sector {
  id: number;
  name: string;
  display_name: string;
  description: string | null;
  icon: string | null;
  is_active: boolean;
  created_at: string;
}

export interface DashboardStats {
  total_orders: number;
  total_revenue: number;
  orders_today: number;
  revenue_today: number;
  orders_by_status: Record<string, number>;
  orders_by_sector: Record<string, number>;
  total_users: number;
  active_sectors: number;
}

export interface PaginatedOrders {
  orders: Order[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface PaginatedUsers {
  users: User[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface OrderFilters {
  search?: string;
  status?: string;
  sector?: string;
  date_from?: string;
  date_to?: string;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  page?: number;
  page_size?: number;
}
