import { useState } from "react";
import type { RewardDistributionData } from "../api/types";
import { RewardCurveChart } from "./RewardCurveChart";

function fmt(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

function DistributionBar({
  bins,
  binEdges,
  binWidth,
  total,
}: {
  bins: number[];
  binEdges: number[];
  binWidth: number;
  total: number;
}) {
  const binCount = bins.length;
  return (
    <div
      className="reward-bar"
      role="img"
      aria-label="reward distribution for this generation"
    >
      {bins.map((c, i) => {
        if (c === 0) return null;
        const lo = binEdges[i];
        const hi = lo + binWidth;
        const widthPct = (c / total) * 100;
        // Color on a red -> green scale by bin position: lower rewards are
        // red, higher rewards green, so the bar's color reads as reward value.
        const f = binCount > 1 ? i / (binCount - 1) : 1;
        const color = `color-mix(in srgb, var(--good) ${Math.round(
          f * 100,
        )}%, var(--danger))`;
        return (
          <div
            key={i}
            className="reward-bar__seg"
            style={{ width: `${widthPct}%`, background: color }}
            title={`[${fmt(lo)}, ${fmt(hi)}): ${c} run${
              c === 1 ? "" : "s"
            } (${widthPct.toFixed(0)}%)`}
          />
        );
      })}
    </div>
  );
}

function DistributionTable({ dist }: { dist: RewardDistributionData }) {
  return (
    <table className="table table--dense">
      <thead>
        <tr>
          <th className="num">Gen</th>
          <th className="num">Runs</th>
          <th className="num">Mean</th>
          <th className="num">Std</th>
          <th className="num">Min</th>
          <th className="num">Max</th>
          <th>Distribution</th>
        </tr>
      </thead>
      <tbody>
        {dist.rows.map((row) => (
          <tr key={row.generation}>
            <td className="num">{row.generation}</td>
            <td className="num">{row.count}</td>
            <td className="num">{row.mean.toFixed(3)}</td>
            <td className="num">{row.std.toFixed(3)}</td>
            <td className="num">{row.min.toFixed(3)}</td>
            <td className="num">{row.max.toFixed(3)}</td>
            <td>
              <DistributionBar
                bins={row.bins}
                binEdges={dist.bin_edges}
                binWidth={dist.bin_width}
                total={row.count}
              />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

type Tab = "table" | "curve";

export function RewardDistribution({
  dist,
}: {
  dist: RewardDistributionData | undefined;
}) {
  const [tab, setTab] = useState<Tab>("table");

  if (!dist || dist.total === 0) {
    return (
      <div className="placeholder placeholder--inset">
        No rewarded runs yet — reward distribution will appear once runs are
        scored.
      </div>
    );
  }

  return (
    <div className="reward-dist">
      <div className="reward-dist__legend">
        <span className="muted">
          {dist.total} rewarded run{dist.total === 1 ? "" : "s"} across{" "}
          {dist.rows.length} generation{dist.rows.length === 1 ? "" : "s"}
        </span>
        <span className="reward-dist__scale" title="bar color encodes reward">
          <span className="muted num">{dist.reward_min.toFixed(2)}</span>
          <span className="reward-dist__gradient" aria-hidden="true" />
          <span className="muted num">{dist.reward_max.toFixed(2)}</span>
        </span>
      </div>

      <div className="tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "table"}
          className={`tab${tab === "table" ? " tab--active" : ""}`}
          onClick={() => setTab("table")}
        >
          Table
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "curve"}
          className={`tab${tab === "curve" ? " tab--active" : ""}`}
          onClick={() => setTab("curve")}
        >
          Curve
        </button>
      </div>

      {tab === "table" ? (
        <DistributionTable dist={dist} />
      ) : (
        <RewardCurveChart rows={dist.rows} />
      )}
    </div>
  );
}
