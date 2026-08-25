interface CategorySelectProps {
  categories: string[];
  value: string;
  onChange: (category: string) => void;
  disabled?: boolean;
}

export function CategorySelect({
  categories,
  value,
  onChange,
  disabled,
}: CategorySelectProps) {
  return (
    <label className="field">
      <span className="field__label">Category</span>
      <select
        className="field__control"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        {categories.map((category) => (
          <option key={category} value={category}>
            {category}
          </option>
        ))}
      </select>
    </label>
  );
}
