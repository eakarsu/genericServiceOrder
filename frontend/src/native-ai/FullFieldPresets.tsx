'use client';

import { type CSSProperties } from 'react';

type Choice = readonly unknown[];
export type PresetChoices = Record<string, Choice>;
export interface FullFieldPreset {
  label: string;
  values: Record<string, unknown>;
}

/** Fill every value (required and optional) with a coherent example. */
export function fillEveryField<V extends Record<string, unknown>>(values: V, variant = 0, choices: PresetChoices = {}): V {
  const names = ['Concise example', 'Planning example', 'Detailed example'];
  const filled = Object.fromEntries(Object.entries(values).map(([key, value]) => {
    if (['id', 'created_at', 'updated_at', 'user_id'].includes(key)) return [key, value];
    const choice = choices[key];
    if (choice && choice.length) return [key, choice[variant % choice.length]];
    if (Array.isArray(value)) return [key, [`${names[variant]} item one`, `${names[variant]} item two`]];
    if (value && typeof value === 'object') return [key, { summary: `${names[variant]} detail`, priority: 'medium' }];
    if (typeof value === 'boolean') return [key, true];
    if (/date|deadline|valid_until/.test(key)) return [key, new Date(Date.now() + (variant + 7) * 86400000).toISOString().slice(0, 10)];
    if (/email/.test(key)) return [key, `reviewer${variant + 1}@example.com`];
    if (/budget|price|cost|amount|micros|cents/.test(key)) return [key, [100, 250, 500][variant]];
    if (/status|state/.test(key)) return [key, ['Draft', 'In review', 'Approved'][variant]];
    if (/title|name|subject/.test(key)) return [key, names[variant]];
    if (typeof value === 'number') return [key, [3, 5, 8][variant]];
    return [key, `${names[variant]}: describe the complete request, include only verified facts, and flag anything that needs an authorized reviewer before it is acted on.`];
  }));
  return filled as V;
}

interface Props {
  value?: Record<string, unknown>;
  onChange?: (values: Record<string, unknown>) => void;
  choices?: PresetChoices;
  presets?: FullFieldPreset[];
  disabled?: boolean;
  heading?: string;
  hint?: string;
}

export default function FullFieldPresets({ value = {}, onChange, choices = {}, presets, disabled = false, heading = 'Fill every field', hint = 'Each example fills the required and optional fields. Review and edit before generating.' }: Props) {
  const labels = presets ? presets.map((preset) => preset.label) : ['Concise example', 'Planning example', 'Detailed example'];
  const apply = (index: number) => {
    const next = presets ? { ...value, ...presets[index].values } : (fillEveryField(value, index, choices) as Record<string, unknown>);
    onChange?.(next);
  };
  return (
    <div className="ai-preset-bar" style={styles.bar}>
      <div style={styles.copy}>
        <strong style={styles.heading}>{heading}</strong>
        <p style={styles.hint}>{hint}</p>
      </div>
      <div className="ai-actions" style={styles.actions}>
        {labels.map((label, index) => (
          <button type="button" className="btn btn-secondary" key={label} disabled={disabled} onClick={() => apply(index)} style={{ ...styles.button, ...(disabled ? styles.buttonDisabled : {}) }}>{label}</button>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  bar: { display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'flex-start', justifyContent: 'space-between', border: '1px solid #dbe3ee', borderRadius: '12px', padding: '12px 14px', background: '#f7fafd', marginBottom: '12px' },
  copy: { minWidth: '220px', flex: '1 1 260px' },
  heading: { display: 'block', fontSize: '13px', color: '#16243a' },
  hint: { margin: '4px 0 0', fontSize: '12px', lineHeight: 1.5, color: '#5b6f8a' },
  actions: { display: 'flex', flexWrap: 'wrap', gap: '8px' },
  button: { border: '1px solid #cdd9e8', background: '#fff', borderRadius: '9px', padding: '7px 12px', fontSize: '12px', fontWeight: 600, color: '#25364d', cursor: 'pointer' },
  buttonDisabled: { opacity: 0.6, cursor: 'not-allowed' },
};
