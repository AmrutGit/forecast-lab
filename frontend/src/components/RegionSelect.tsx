interface RegionSelectProps {
  regions: string[];
  value: string;
  onChange: (region: string) => void;
  disabled?: boolean;
}

export function RegionSelect({ regions, value, onChange, disabled }: RegionSelectProps) {
  return (
    <label className="field">
      <span className="field__label">Region</span>
      <select
        className="field__control"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        {regions.map((region) => (
          <option key={region} value={region}>
            {region}
          </option>
        ))}
      </select>
    </label>
  );
}
