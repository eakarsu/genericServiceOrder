import { ORDER_STATUSES } from '../../utils/constants';
import type { Sector } from '../../types';

interface FilterControlsProps {
  status: string;
  sector: string;
  dateFrom: string;
  dateTo: string;
  sectors: Sector[];
  onStatusChange: (v: string) => void;
  onSectorChange: (v: string) => void;
  onDateFromChange: (v: string) => void;
  onDateToChange: (v: string) => void;
  onClear: () => void;
}

export default function FilterControls({
  status, sector, dateFrom, dateTo, sectors,
  onStatusChange, onSectorChange, onDateFromChange, onDateToChange, onClear,
}: FilterControlsProps) {
  const hasFilters = status || sector || dateFrom || dateTo;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <select value={status} onChange={e => onStatusChange(e.target.value)}
        className="text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <option value="">All Statuses</option>
        {ORDER_STATUSES.map(s => (
          <option key={s.value} value={s.value}>{s.label}</option>
        ))}
      </select>
      <select value={sector} onChange={e => onSectorChange(e.target.value)}
        className="text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <option value="">All Sectors</option>
        {sectors.map(s => (
          <option key={s.name} value={s.name}>{s.display_name}</option>
        ))}
      </select>
      <input type="date" value={dateFrom} onChange={e => onDateFromChange(e.target.value)}
        className="text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
      <input type="date" value={dateTo} onChange={e => onDateToChange(e.target.value)}
        className="text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
      {hasFilters && (
        <button onClick={onClear} className="text-sm text-blue-600 hover:text-blue-800">
          Clear filters
        </button>
      )}
    </div>
  );
}
