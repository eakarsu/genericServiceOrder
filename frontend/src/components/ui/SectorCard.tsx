import { useNavigate } from 'react-router-dom';
import { SECTOR_ICONS } from '../../utils/constants';
import type { Sector } from '../../types';

interface SectorCardProps {
  sector: Sector;
  orderCount?: number;
}

export default function SectorCard({ sector, orderCount }: SectorCardProps) {
  const navigate = useNavigate();
  const emoji = SECTOR_ICONS[sector.icon || ''] || '📦';

  return (
    <div
      onClick={() => navigate(`/sectors/${sector.name}`)}
      className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 hover:shadow-md hover:border-blue-200 cursor-pointer transition-all"
    >
      <div className="text-3xl mb-3">{emoji}</div>
      <h3 className="font-semibold text-gray-900">{sector.display_name}</h3>
      <p className="text-sm text-gray-500 mt-1 line-clamp-2">{sector.description}</p>
      {orderCount !== undefined && (
        <p className="text-xs text-gray-400 mt-2">{orderCount} orders</p>
      )}
    </div>
  );
}
