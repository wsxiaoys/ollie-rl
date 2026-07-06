import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
} from "chart.js";
import { Line } from "react-chartjs-2";
import type { GenerationRewardStats } from "../api/types";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
);

/** Read a CSS custom property off the document root, with a fallback. */
function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return v || fallback;
}

/**
 * Reward curve over policy generations: a mean line with a shaded min/max
 * envelope, so the reward trend and its spread across training are visible at
 * a glance.
 */
export function RewardCurveChart({ rows }: { rows: GenerationRewardStats[] }) {
  const accent = cssVar("--accent", "#58a6ff");
  const good = cssVar("--good", "#3fb950");
  const muted = cssVar("--text-muted", "#8b949e");
  const border = cssVar("--border", "#2a313c");
  const band = "color-mix(in srgb, " + accent + " 18%, transparent)";

  const labels = rows.map((r) => String(r.generation));

  const data: ChartData<"line"> = {
    labels,
    datasets: [
      {
        label: "min",
        data: rows.map((r) => r.min),
        borderColor: muted,
        borderDash: [4, 4],
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        tension: 0.25,
      },
      {
        label: "max",
        data: rows.map((r) => r.max),
        borderColor: muted,
        borderDash: [4, 4],
        borderWidth: 1,
        pointRadius: 0,
        // Fill toward the previous dataset (min) to shade the spread band.
        fill: "-1",
        backgroundColor: band,
        tension: 0.25,
      },
      {
        label: "mean",
        data: rows.map((r) => r.mean),
        borderColor: accent,
        backgroundColor: accent,
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: good,
        fill: false,
        tension: 0.25,
      },
    ],
  };

  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: {
        title: { display: true, text: "policy generation", color: muted },
        ticks: { color: muted },
        grid: { color: border },
      },
      y: {
        title: { display: true, text: "reward", color: muted },
        ticks: { color: muted },
        grid: { color: border },
      },
    },
    plugins: {
      legend: { labels: { color: muted } },
      tooltip: {
        callbacks: {
          afterBody: (items) => {
            const idx = items[0]?.dataIndex;
            if (idx === undefined) return "";
            const row = rows[idx];
            return `std ${row.std.toFixed(3)} · ${row.count} run${
              row.count === 1 ? "" : "s"
            }`;
          },
        },
      },
    },
  };

  return (
    <div className="reward-curve">
      <Line data={data} options={options} />
    </div>
  );
}
