import { useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import type { DatumProgress } from "../api/types";
import { Badge, ProgressBar } from "./ui";

const columnHelper = createColumnHelper<DatumProgress>();

export type QuarantineConfig = {
  quarantineMinSamples: number;
  maxUnhealthyFinishRatio: number;
  maxSucceedRatio: number;
};

export type QuarantineState = {
  unhealthyRate: number | null;
  succeedRate: number | null;
  belowMinSamples: boolean;
  unhealthyHit: boolean;
  succeedHit: boolean;
  quarantined: boolean;
};

/**
 * Mirror the dispenser's quarantine logic for a single datum. Both filters
 * share the full `rewarded` denominator and the `min_samples` gate; every
 * rewarded run is a sample (content-filtered/malformed runs included):
 *   - unhealthy-finish rate = (length + content_filter) / rewarded
 *   - success ratio         = succeeded / rewarded
 * `length` (length-limited) and `content_filter` (malformed) are both
 * auto-penalty degenerate rollouts, so they're summed into one unhealthy-finish
 * numerator. Expired/lost runs are status observability only and do not affect
 * quarantine rates. Below the min-sample gate neither filter can fire.
 */
export function computeQuarantine(
  item: Pick<
    DatumProgress,
    "length" | "rewarded" | "succeeded" | "content_filter"
  >,
  {
    quarantineMinSamples,
    maxUnhealthyFinishRatio,
    maxSucceedRatio,
  }: QuarantineConfig,
): QuarantineState {
  const { length, rewarded, succeeded, content_filter } = item;
  // Unhealthy finishes (length-limited + malformed) share one numerator; both
  // filters divide by the full `rewarded` count.
  const unhealthyRate = rewarded > 0 ? (length + content_filter) / rewarded : null;
  const succeedRate = rewarded > 0 ? succeeded / rewarded : null;
  const belowMinSamples = rewarded < quarantineMinSamples;
  const unhealthyHit =
    !belowMinSamples &&
    unhealthyRate != null &&
    unhealthyRate >= maxUnhealthyFinishRatio;
  const succeedHit =
    !belowMinSamples && succeedRate != null && succeedRate >= maxSucceedRatio;
  return {
    unhealthyRate,
    succeedRate,
    belowMinSamples,
    unhealthyHit,
    succeedHit,
    quarantined: unhealthyHit || succeedHit,
  };
}

export function DatumTable({
  items,
  groupSize,
  quarantineMinSamples,
  maxUnhealthyFinishRatio,
  maxSucceedRatio,
  hideExcluded = false,
  tunerId,
}: {
  items: DatumProgress[];
  groupSize: number;
  quarantineMinSamples: number;
  maxUnhealthyFinishRatio: number;
  maxSucceedRatio: number;
  hideExcluded?: boolean;
  tunerId: string;
}) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "consumable", desc: true },
  ]);

  // react-table requires `data` and `columns` to be referentially stable
  // between renders. This component re-renders every couple of seconds (the
  // tuner query polls on an interval), so recreating the filtered array/columns
  // inline would hand react-table a fresh reference each time and thrash the
  // row model — which manifests as the table locking up when excluded rows are
  // hidden. Memoize both so the references only change when their inputs do.
  const columns = useMemo(() => [
    columnHelper.accessor("datum_id", {
      header: "Datum ID",
      cell: (info) => (
        <Link
          to="/datums"
          search={{ tuner: tunerId, datum: info.getValue() }}
          className="mono link-plain"
        >
          {info.getValue()}
        </Link>
      ),
    }),
    columnHelper.accessor("consumable", {
      header: `Consumable / ${groupSize}`,
      cell: (info) => {
        const v = info.getValue();
        return (
          <div className="datum-progress">
            <ProgressBar value={v} max={groupSize} tone="good" />
            <span className="datum-progress__label">
              {v}/{groupSize}
            </span>
          </div>
        );
      },
    }),
    columnHelper.accessor("in_flight", {
      header: "In flight",
      cell: (info) => <span className="num">{info.getValue()}</span>,
    }),
    columnHelper.accessor("length", {
      id: "audit",
      header: "Audit",
      cell: (info) => {
        const length = info.row.original.length;
        const rewarded = info.row.original.rewarded;
        const succeeded = info.row.original.succeeded;
        const contentFilter = info.row.original.content_filter;
        // Unhealthy finishes = length-limited + content-filtered (malformed).
        const unhealthy = length + contentFilter;
        const minSamples = quarantineMinSamples;
        const pct = (r: number) => `${(r * 100).toFixed(0)}%`;
        const {
          unhealthyRate,
          succeedRate,
          belowMinSamples,
          unhealthyHit,
          succeedHit,
          quarantined,
        } = computeQuarantine(info.row.original, {
          quarantineMinSamples,
          maxUnhealthyFinishRatio,
          maxSucceedRatio,
        });
        const definition =
          'A "length" run is a rewarded run with at least one completion whose finish_reason is "length"; a "content_filter" run is a malformed rewarded run. Both are "unhealthy" finishes — auto-penalty degenerate rollouts with no verifier grade. A "succeeded" run earned reward == 1.0.\n' +
          "Both filters divide by the full rewarded count:\n" +
          "• Unhealthy-finish rate = (length + content_filter) / rewarded.\n" +
          "• Succeed ratio = succeeded / rewarded.";
        const quarantineReason = unhealthyHit
          ? `Quarantined: unhealthy-finish rate ${pct(unhealthyRate!)} ≥ max_unhealthy_finish_ratio ${pct(maxUnhealthyFinishRatio)} — no new runs dispensed.\n`
          : succeedHit
            ? `Quarantined: succeed ratio ${pct(succeedRate!)} ≥ max_succeed_ratio ${pct(maxSucceedRatio)} — no new runs dispensed.\n`
            : "";
        const unhealthyNote =
          unhealthy > 0
            ? `Unhealthy finishes: ${length} length + ${contentFilter} content_filter = ${unhealthy} of ${rewarded} rewarded.\n`
            : "";
        const tooltip =
          unhealthyRate == null || succeedRate == null
            ? `No rewarded attempts yet to compute ratios.\n${definition}`
            : `${rewarded} rewarded attempts.\n` +
              unhealthyNote +
              (belowMinSamples
                ? `Below the ${minSamples} min-sample gate — ratios shown but not yet actionable for quarantine.\n`
                : "") +
              quarantineReason +
              `Unhealthy-finish ratio ${pct(unhealthyRate)}: ${unhealthy} unhealthy (${length} length + ${contentFilter} content_filter) / ${rewarded} rewarded — drives max_unhealthy_finish_ratio (${pct(maxUnhealthyFinishRatio)}).\n` +
              `Succeed ratio ${pct(succeedRate)}: ${succeeded} succeeded / ${rewarded} rewarded — drives max_succeed_ratio (${pct(maxSucceedRatio)}).\n` +
              definition;
        if (unhealthyRate == null || succeedRate == null) {
          return (
            <span className="num muted" title={tooltip}>
              —
            </span>
          );
        }
        const ratios = (
          <>
            <span className={unhealthyHit ? "datum-audit__hit" : undefined}>
              {pct(unhealthyRate)}
              <span className="muted"> unhealthy</span>
            </span>
            {" · "}
            <span className={succeedHit ? "datum-audit__hit" : undefined}>
              {pct(succeedRate)}
              <span className="muted"> ok</span>
            </span>
          </>
        );
        // Below the min-sample gate the ratios aren't actionable yet. Colour
        // (muted text) alone isn't distinct enough, so add non-colour cues:
        // wrap the ratios in parentheses and append an explicit sample marker
        // (`<rewarded>/<minSamples>`) showing how far off the gate we are.
        if (belowMinSamples) {
          return (
            <span className="num muted" title={tooltip}>
              ({ratios}){" "}
              <span className="datum-audit__gate">
                {rewarded}/{minSamples}
              </span>
            </span>
          );
        }
        return (
          <span className="num" title={tooltip}>
            {ratios}
            {quarantined && (
              <span className="datum-audit__quarantine">
                <Badge tone="warn" title={quarantineReason.trim()}>
                  excluded
                </Badge>
              </span>
            )}
          </span>
        );
      },
    }),
    columnHelper.accessor("trained", {
      header: "Trained",
      cell: (info) => <span className="num">{info.getValue()}</span>,
    }),
  ], [
    groupSize,
    tunerId,
    quarantineMinSamples,
    maxUnhealthyFinishRatio,
    maxSucceedRatio,
  ]);

  const visibleItems = useMemo(
    () =>
      hideExcluded
        ? items.filter(
            (item) =>
              !computeQuarantine(item, {
                quarantineMinSamples,
                maxUnhealthyFinishRatio,
                maxSucceedRatio,
              }).quarantined,
          )
        : items,
    [
      items,
      hideExcluded,
      quarantineMinSamples,
      maxUnhealthyFinishRatio,
      maxSucceedRatio,
    ],
  );

  const table = useReactTable({
    data: visibleItems,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (items.length === 0) {
    return (
      <div className="placeholder placeholder--inset">
        No data is currently in progress.
      </div>
    );
  }

  if (visibleItems.length === 0) {
    return (
      <div className="placeholder placeholder--inset">
        All data is currently excluded. Toggle “Hide excluded” to view it.
      </div>
    );
  }

  return (
    <table className="table table--dense">
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id}>
            {hg.headers.map((header) => (
              <th
                key={header.id}
                onClick={header.column.getToggleSortingHandler()}
                className="sortable"
              >
                {flexRender(
                  header.column.columnDef.header,
                  header.getContext(),
                )}
                {{ asc: " ▲", desc: " ▼" }[
                  header.column.getIsSorted() as string
                ] ?? ""}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
