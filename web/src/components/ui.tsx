import type { ReactNode } from "react";

export function StatCard({
  label,
  value,
  tone = "default",
  title,
}: {
  label: string;
  value: ReactNode;
  tone?: "default" | "good" | "warn" | "muted" | "danger";
  title?: string;
}) {
  return (
    <div className={`stat-card stat-card--${tone}`} title={title}>
      <div className="stat-card__value">{value}</div>
      <div className="stat-card__label">{label}</div>
    </div>
  );
}

export function ProgressBar({
  value,
  max,
  tone = "default",
}: {
  value: number;
  max: number;
  tone?: "default" | "good";
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="progress">
      <div
        className={`progress__fill progress__fill--${tone}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function Badge({
  children,
  tone = "default",
  title,
}: {
  children: ReactNode;
  tone?: "default" | "good" | "warn" | "danger" | "info";
  title?: string;
}) {
  return (
    <span className={`badge badge--${tone}`} title={title}>
      {children}
    </span>
  );
}

export function Panel({
  title,
  children,
  right,
}: {
  title: string;
  children: ReactNode;
  right?: ReactNode;
}) {
  return (
    <section className="panel">
      <header className="panel__header">
        <h2 className="panel__title">{title}</h2>
        {right}
      </header>
      <div className="panel__body">{children}</div>
    </section>
  );
}

export function Mono({ children }: { children: ReactNode }) {
  return <span className="mono">{children}</span>;
}
