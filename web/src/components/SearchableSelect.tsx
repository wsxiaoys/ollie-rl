import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

/**
 * A combobox: a button that opens a popup with a quick-search box and a
 * filtered list of options. Selecting an option (or clearing) reports the new
 * value to the parent. The current value is fully controlled by `value`.
 */
export function SearchableSelect({
  value,
  options,
  onChange,
  placeholder = "Select…",
  searchPlaceholder = "Search…",
  disabled = false,
  clearable = true,
  id,
}: {
  value: string | null;
  options: string[];
  onChange: (value: string | null) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  disabled?: boolean;
  clearable?: boolean;
  id?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter((o) => o.toLowerCase().includes(q));
  }, [options, query]);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  // Reset the search box and focus it whenever the popup opens.
  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      // Focus after paint so the input exists.
      requestAnimationFrame(() => searchRef.current?.focus());
    }
  }, [open]);

  // Keep the highlighted row within the filtered range.
  useEffect(() => {
    setActiveIndex((i) => Math.min(i, Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

  const select = (v: string) => {
    onChange(v);
    setOpen(false);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const picked = filtered[activeIndex];
      if (picked) select(picked);
    } else if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
    }
  };

  return (
    <div className="combo" ref={rootRef}>
      <button
        id={id}
        type="button"
        className="combo__control"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => !disabled && setOpen((o) => !o)}
      >
        <span className={value ? "combo__value" : "combo__placeholder"}>
          {value ?? placeholder}
        </span>
        <span className="combo__chevron" aria-hidden>
          ▾
        </span>
      </button>

      {clearable && value && !disabled && (
        <button
          type="button"
          className="combo__clear"
          aria-label="Clear selection"
          onClick={() => onChange(null)}
        >
          ×
        </button>
      )}

      {open && (
        <div className="combo__popup" role="dialog">
          <input
            ref={searchRef}
            type="search"
            className="combo__search"
            placeholder={searchPlaceholder}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <ul className="combo__list" role="listbox">
            {filtered.length === 0 && (
              <li className="combo__empty">No matches</li>
            )}
            {filtered.map((opt, i) => (
              <li
                key={opt}
                role="option"
                aria-selected={opt === value}
                className={
                  "combo__option" +
                  (i === activeIndex ? " combo__option--active" : "") +
                  (opt === value ? " combo__option--selected" : "")
                }
                onMouseEnter={() => setActiveIndex(i)}
                onMouseDown={(e) => {
                  // Prevent the search input from losing focus before click.
                  e.preventDefault();
                  select(opt);
                }}
              >
                {opt}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
