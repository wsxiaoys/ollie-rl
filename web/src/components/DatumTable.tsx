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
  maxLengthRatio: number;
  maxSucceedRatio: number;
};

export type QuarantineState = {
  lengthRate: number | null;
  succeedRate: number | null;
  belowMinSamples: boolean;
  lengthHit: boolean;
  succeedHit: boolean;
  quarantined: boolean;
};

/**
 * Mirror the dispenser's quarantine logic for a single datum. Both quarantine
 * metrics share the rewarded-attempt denominator; expired/lost runs are status
 * observability only and do not affect quarantine rates. Below the min-sample
 * gate neither filter can fire, so the ratios are not yet actionable.
 */
export function computeQuarantine(
  item: Pick<DatumProgress, "length" | "rewarded" | "succeeded">,
  { quarantineMinSamples, maxLengthRatio, maxSucceedRatio }: QuarantineConfig,
): QuarantineState {
  const { length, rewarded, succeeded } = item;
  const lengthRate = rewarded > 0 ? length / rewarded : null;
  const succeedRate = rewarded > 0 ? succeeded / rewarded : null;
  const belowMinSamples = rewarded < quarantineMinSamples;
  const lengthHit =
    !belowMinSamples && lengthRate != null && lengthRate >= maxLengthRatio;
  const succeedHit =
    !belowMinSamples && succeedRate != null && succeedRate >= maxSucceedRatio;
  return {
    lengthRate,
    succeedRate,
    belowMinSamples,
    lengthHit,
    succeedHit,
    quarantined: lengthHit || succeedHit,
  };
}

export function DatumTable({
  items,
  groupSize,
  quarantineMinSamples,
  maxLengthRatio,
  maxSucceedRatio,
  hideExcluded = false,
  tunerId,
}: {
  items: DatumProgress[];
  groupSize: number;
  quarantineMinSamples: number;
  maxLengthRatio: number;
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
        const minSamples = quarantineMinSamples;
        const pct = (r: number) => `${(r * 100).toFixed(0)}%`;
        const {
          lengthRate,
          succeedRate,
          belowMinSamples,
          lengthHit,
          succeedHit,
          quarantined,
        } = computeQuarantine(info.row.original, {
          quarantineMinSamples,
          maxLengthRatio,
          maxSucceedRatio,
        });
        const definition =
          'A "length" run is a rewarded run with at least one completion whose finish_reason is "length". A "succeeded" run earned reward == 1.0.';
        const quarantineReason = lengthHit
          ? `Quarantined: length rate ${pct(lengthRate!)} ≥ max_length_ratio ${pct(maxLengthRatio)} — no new runs dispensed.\n`
          : succeedHit
            ? `Quarantined: succeed ratio ${pct(succeedRate!)} ≥ max_succeed_ratio ${pct(maxSucceedRatio)} — no new runs dispensed.\n`
            : "";
        const tooltip =
          lengthRate == null || succeedRate == null
            ? `No rewarded attempts yet to compute ratios.\n${definition}`
            : `${rewarded} rewarded attempts.\n` +
              (belowMinSamples
                ? `Below the ${minSamples} min-sample gate — ratios shown but not yet actionable for quarantine.\n`
                : "") +
              quarantineReason +
              `Length ratio ${pct(lengthRate)}: ${length} length / ${rewarded} — drives max_length_ratio (${pct(maxLengthRatio)}).\n` +
              `Succeed ratio ${pct(succeedRate)}: ${succeeded} succeeded / ${rewarded} — drives max_succeed_ratio (${pct(maxSucceedRatio)}).\n` +
              definition;
        if (lengthRate == null || succeedRate == null) {
          return (
            <span className="num muted" title={tooltip}>
              —
            </span>
          );
        }
        const ratios = (
          <>
            <span className={lengthHit ? "datum-audit__hit" : undefined}>
              {pct(lengthRate)}
              <span className="muted"> len</span>
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
        // (`n <rewarded>/<minSamples>`) showing how far off the gate we are.
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
  ], [groupSize, tunerId, quarantineMinSamples, maxLengthRatio, maxSucceedRatio]);

  const visibleItems = useMemo(
    () =>
      hideExcluded
        ? items.filter(
            (item) =>
              !computeQuarantine(item, {
                quarantineMinSamples,
                maxLengthRatio,
                maxSucceedRatio,
              }).quarantined,
          )
        : items,
    [items, hideExcluded, quarantineMinSamples, maxLengthRatio, maxSucceedRatio],
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
