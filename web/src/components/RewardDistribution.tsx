import { useMemo, useState } from "react";
import type { RunItem } from "../api/types";
import { RewardCurveChart } from "./RewardCurveChart";

const BIN_COUNT = 12;

export type GenStats = {
  generation: number;
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  /** Per-bin reward counts, aligned to the shared global bin edges. */
  bins: number[];
};

export type Distribution = {
  /** Per-generation rows, ascending by generation. */
  rows: GenStats[];
  /** Shared lower edges of each histogram bin (length BIN_COUNT). */
  binEdges: number[];
  binWidth: number;
  /** Global reward range across all rewarded runs. */
  rewardMin: number;
  rewardMax: number;
  /** Total rewarded runs that contributed (reward + generation present). */
  total: number;
};

function computeDistribution(runs: RunItem[]): Distribution | null {
  // A run only contributes if it has both a reward and a derived policy
  // generation (max generation across its completions). Runs still awaiting a
  // reward, or without any recorded completion, are excluded.
  const rewarded = runs.filter(
    (r): r is RunItem & { reward: number; policy_generation: number } =>
      r.reward !== null && r.policy_generation !== null,
  );

  if (rewarded.length === 0) return null;

  let rewardMin = Number.POSITIVE_INFINITY;
  let rewardMax = Number.NEGATIVE_INFINITY;
  for (const r of rewarded) {
    if (r.reward < rewardMin) rewardMin = r.reward;
    if (r.reward > rewardMax) rewardMax = r.reward;
  }

  // Guard against a degenerate range (all rewards equal): widen by a unit so
  // every value lands in a valid bin instead of dividing by zero.
  const span = rewardMax - rewardMin;
  const effectiveSpan = span > 0 ? span : 1;
  const binWidth = effectiveSpan / BIN_COUNT;
  const binEdges = Array.from(
    { length: BIN_COUNT },
    (_, i) => rewardMin + i * binWidth,
  );

  const binOf = (reward: number) => {
    const idx = Math.floor((reward - rewardMin) / binWidth);
    // Clamp so the maximum reward falls into the last bin instead of overflowing.
    return Math.min(BIN_COUNT - 1, Math.max(0, idx));
  };

  const byGeneration = new Map<number, number[]>();
  for (const r of rewarded) {
    const list = byGeneration.get(r.policy_generation);
    if (list) {
      list.push(r.reward);
    } else {
      byGeneration.set(r.policy_generation, [r.reward]);
    }
  }

  const rows: GenStats[] = [];
  for (const [generation, rewards] of byGeneration) {
    const count = rewards.length;
    const mean = rewards.reduce((a, b) => a + b, 0) / count;
    const variance =
      rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / count;
    const std = Math.sqrt(variance);
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    const bins = new Array<number>(BIN_COUNT).fill(0);
    for (const reward of rewards) {
      if (reward < min) min = reward;
      if (reward > max) max = reward;
      bins[binOf(reward)] += 1;
    }
    rows.push({ generation, count, mean, std, min, max, bins });
  }

  rows.sort((a, b) => a.generation - b.generation);

  return {
    rows,
    binEdges,
    binWidth,
    rewardMin,
    rewardMax,
    total: rewarded.length,
  };
}

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
        const f = BIN_COUNT > 1 ? i / (BIN_COUNT - 1) : 1;
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

function DistributionTable({ dist }: { dist: Distribution }) {
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
                binEdges={dist.binEdges}
                binWidth={dist.binWidth}
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

export function RewardDistribution({ runs }: { runs: RunItem[] }) {
  const dist = useMemo(() => computeDistribution(runs), [runs]);
  const [tab, setTab] = useState<Tab>("table");

  if (!dist) {
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
          <span className="muted num">{dist.rewardMin.toFixed(2)}</span>
          <span className="reward-dist__gradient" aria-hidden="true" />
          <span className="muted num">{dist.rewardMax.toFixed(2)}</span>
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
