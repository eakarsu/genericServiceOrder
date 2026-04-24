import { useState, useEffect, useCallback } from 'react';
import { Plus, Edit2, Trash2, X, Save } from 'lucide-react';
import { getUsers, createUser, updateUser, deleteUser } from '../api/users';
import type { User } from '../types';
import { useToast } from '../context/ToastContext';
import SearchBar from '../components/ui/SearchBar';
import Pagination from '../components/ui/Pagination';
import EmptyState from '../components/ui/EmptyState';
import ConfirmDialog from '../components/ui/ConfirmDialog';
import { SkeletonTable } from '../components/ui/SkeletonLoader';
import { useDebounce } from '../hooks/useDebounce';
import { formatDate } from '../utils/formatters';

const ROLES = [
  { id: 1, name: 'admin', label: 'Admin' },
  { id: 2, name: 'manager', label: 'Manager' },
  { id: 3, name: 'viewer', label: 'Viewer' },
];

export default function UsersPage() {
  const { addToast } = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const debouncedSearch = useDebounce(search);

  const [showCreate, setShowCreate] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<number | null>(null);

  const [formData, setFormData] = useState({ name: '', email: '', password: '', phone: '', role_id: 3 });
  const [editData, setEditData] = useState({ name: '', phone: '', role_id: 3, is_active: true });

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getUsers({ page, page_size: 20, search: debouncedSearch || undefined });
      setUsers(data.users);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch {
      addToast('Failed to load users', 'error');
    } finally {
      setLoading(false);
    }
  }, [page, debouncedSearch, addToast]);

  useEffect(() => { fetchUsers(); }, [fetchUsers]);

  const handleCreate = async () => {
    try {
      await createUser(formData);
      addToast('User created', 'success');
      setShowCreate(false);
      setFormData({ name: '', email: '', password: '', phone: '', role_id: 3 });
      fetchUsers();
    } catch (err: any) {
      addToast(err.response?.data?.detail || 'Failed to create user', 'error');
    }
  };

  const handleEdit = (u: User) => {
    setEditingUser(u);
    setEditData({ name: u.name, phone: u.phone || '', role_id: u.role_id, is_active: u.is_active });
  };

  const handleSaveEdit = async () => {
    if (!editingUser) return;
    try {
      await updateUser(editingUser.id, editData);
      addToast('User updated', 'success');
      setEditingUser(null);
      fetchUsers();
    } catch (err: any) {
      addToast(err.response?.data?.detail || 'Failed to update', 'error');
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteUser(id);
      addToast('User deleted', 'success');
      setDeleteTarget(null);
      fetchUsers();
    } catch (err: any) {
      addToast(err.response?.data?.detail || 'Failed to delete', 'error');
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Users</h1>
        <button onClick={() => setShowCreate(true)} className="flex items-center gap-1 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">
          <Plus size={16} /> Add User
        </button>
      </div>

      <div className="w-full sm:w-72">
        <SearchBar value={search} onChange={v => { setSearch(v); setPage(1); }} placeholder="Search users..." />
      </div>

      <div className="bg-white rounded-xl shadow-sm border overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              <th className="px-4 py-3 text-left font-medium">Name</th>
              <th className="px-4 py-3 text-left font-medium">Email</th>
              <th className="px-4 py-3 text-left font-medium hidden md:table-cell">Role</th>
              <th className="px-4 py-3 text-left font-medium hidden md:table-cell">Status</th>
              <th className="px-4 py-3 text-left font-medium hidden lg:table-cell">Joined</th>
              <th className="px-4 py-3 text-right font-medium">Actions</th>
            </tr>
          </thead>
          {loading ? (
            <SkeletonTable rows={10} cols={6} />
          ) : users.length === 0 ? (
            <tbody><tr><td colSpan={6}><EmptyState title="No users found" /></td></tr></tbody>
          ) : (
            <tbody className="divide-y divide-gray-100">
              {users.map(u => (
                <tr key={u.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-gray-900">{u.name}</td>
                  <td className="px-4 py-3 text-gray-600">{u.email}</td>
                  <td className="px-4 py-3 hidden md:table-cell">
                    <span className="px-2 py-0.5 bg-gray-100 rounded text-xs capitalize">{u.role_name}</span>
                  </td>
                  <td className="px-4 py-3 hidden md:table-cell">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${u.is_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {u.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                  <td className="px-4 py-3 hidden lg:table-cell text-gray-500">{formatDate(u.created_at)}</td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex justify-end gap-1">
                      <button onClick={() => handleEdit(u)} className="p-1.5 text-gray-400 hover:text-blue-600 rounded"><Edit2 size={14} /></button>
                      <button onClick={() => setDeleteTarget(u.id)} className="p-1.5 text-gray-400 hover:text-red-600 rounded"><Trash2 size={14} /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          )}
        </table>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-500">{total} users total</p>
        <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
      </div>

      {/* Create User Modal */}
      {showCreate && (
        <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/50" onClick={() => setShowCreate(false)}>
          <div className="bg-white rounded-xl shadow-xl p-6 max-w-md w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Create User</h3>
              <button onClick={() => setShowCreate(false)} className="text-gray-400 hover:text-gray-600"><X size={20} /></button>
            </div>
            <div className="space-y-3">
              <input placeholder="Name" value={formData.name} onChange={e => setFormData(p => ({ ...p, name: e.target.value }))}
                className="w-full border rounded-lg px-3 py-2 text-sm" />
              <input placeholder="Email" type="email" value={formData.email} onChange={e => setFormData(p => ({ ...p, email: e.target.value }))}
                className="w-full border rounded-lg px-3 py-2 text-sm" />
              <input placeholder="Password" type="password" value={formData.password} onChange={e => setFormData(p => ({ ...p, password: e.target.value }))}
                className="w-full border rounded-lg px-3 py-2 text-sm" />
              <input placeholder="Phone (optional)" value={formData.phone} onChange={e => setFormData(p => ({ ...p, phone: e.target.value }))}
                className="w-full border rounded-lg px-3 py-2 text-sm" />
              <select value={formData.role_id} onChange={e => setFormData(p => ({ ...p, role_id: Number(e.target.value) }))}
                className="w-full border rounded-lg px-3 py-2 text-sm">
                {ROLES.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
              </select>
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <button onClick={() => setShowCreate(false)} className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200">Cancel</button>
              <button onClick={handleCreate} className="px-4 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700">Create</button>
            </div>
          </div>
        </div>
      )}

      {/* Edit User Modal */}
      {editingUser && (
        <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/50" onClick={() => setEditingUser(null)}>
          <div className="bg-white rounded-xl shadow-xl p-6 max-w-md w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Edit User</h3>
              <button onClick={() => setEditingUser(null)} className="text-gray-400 hover:text-gray-600"><X size={20} /></button>
            </div>
            <div className="space-y-3">
              <input placeholder="Name" value={editData.name} onChange={e => setEditData(p => ({ ...p, name: e.target.value }))}
                className="w-full border rounded-lg px-3 py-2 text-sm" />
              <input placeholder="Phone" value={editData.phone} onChange={e => setEditData(p => ({ ...p, phone: e.target.value }))}
                className="w-full border rounded-lg px-3 py-2 text-sm" />
              <select value={editData.role_id} onChange={e => setEditData(p => ({ ...p, role_id: Number(e.target.value) }))}
                className="w-full border rounded-lg px-3 py-2 text-sm">
                {ROLES.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
              </select>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={editData.is_active} onChange={e => setEditData(p => ({ ...p, is_active: e.target.checked }))} className="rounded" />
                Active
              </label>
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <button onClick={() => setEditingUser(null)} className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200">Cancel</button>
              <button onClick={handleSaveEdit} className="flex items-center gap-1 px-4 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700"><Save size={14} /> Save</button>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete User"
        message="Are you sure you want to delete this user?"
        confirmLabel="Delete"
        onConfirm={() => deleteTarget && handleDelete(deleteTarget)}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
