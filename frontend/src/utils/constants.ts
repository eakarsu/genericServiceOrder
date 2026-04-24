export const ORDER_STATUSES = [
  { value: 'pending', label: 'Pending', color: 'bg-yellow-100 text-yellow-800' },
  { value: 'confirmed', label: 'Confirmed', color: 'bg-blue-100 text-blue-800' },
  { value: 'in_progress', label: 'In Progress', color: 'bg-purple-100 text-purple-800' },
  { value: 'completed', label: 'Completed', color: 'bg-green-100 text-green-800' },
  { value: 'cancelled', label: 'Cancelled', color: 'bg-red-100 text-red-800' },
] as const;

export const SECTOR_ICONS: Record<string, string> = {
  wrench: '🔧', scissors: '✂️', 'graduation-cap': '🎓', calendar: '📅',
  'dollar-sign': '💰', dumbbell: '💪', utensils: '🍴', 'heart-pulse': '❤️',
  home: '🏠', shield: '🛡️', monitor: '💻', shirt: '👔', scale: '⚖️',
  truck: '🚚', 'paw-print': '🐾', camera: '📷', building: '🏢', car: '🚗', plane: '✈️',
};

export const PAGE_SIZES = [10, 20, 50, 100];
