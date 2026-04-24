import api from './client';

interface ExportParams {
  sector?: string;
  status?: string;
  date_from?: string;
  date_to?: string;
  search?: string;
}

export async function exportCSV(params: ExportParams = {}): Promise<Blob> {
  const res = await api.post('/export/csv', params, { responseType: 'blob' });
  return res.data;
}

export async function exportPDF(params: ExportParams = {}): Promise<Blob> {
  const res = await api.post('/export/pdf', params, { responseType: 'blob' });
  return res.data;
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
