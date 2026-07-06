import { useState } from "react";
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
import { ProgressBar } from "./ui";

const columnHelper = createColumnHelper<DatumProgress>();

export function DatumTable({
  items,
  groupSize,
  tunerId,
}: {
  items: DatumProgress[];
  groupSize: number;
  tunerId: string;
}) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "consumable", desc: true },
  ]);

  const columns = [
    columnHelper.accessor("datum_id", {
      header: "Datum ID",
      cell: (info) => (
        <Link
          to="/data"
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
    columnHelper.accessor("expired", {
      id: "audit",
      header: "Audit",
      cell: (info) => {
        const expired = info.row.original.expired;
        const rewarded = info.row.original.rewarded;
        const succeeded = info.row.original.succeeded;
        // Both quarantine metrics share the terminal denominator
        // (expired + rewarded); in-flight runs are excluded (outcome unknown).
        const terminal = expired + rewarded;
        const expireRate = terminal > 0 ? expired / terminal : null;
        const succeedRate = terminal > 0 ? succeeded / terminal : null;
        // Mirror the dispenser's `min_samples = 0.5 * group_size` gate: below it
        // neither quarantine filter can fire, so the ratios are not yet
        // actionable — render them muted to signal "not enough samples".
        const minSamples = groupSize / 2;
        const belowMinSamples = terminal < minSamples;
        const pct = (r: number) => `${(r * 100).toFixed(0)}%`;
        const definition =
          'An "expired" run is an expired, unrewarded run whose generation op is still in flight — i.e. the generation itself stalled past the lease, rather than a "lost" (crashed/abandoned) worker. A "succeeded" run earned reward == 1.0.';
        const tooltip =
          expireRate == null || succeedRate == null
            ? `No terminal attempts yet to compute ratios.\n${definition}`
            : `${terminal} terminal attempts (expired + rewarded).\n` +
              (belowMinSamples
                ? `Below the ${minSamples} min-sample gate — ratios shown but not yet actionable for quarantine.\n`
                : "") +
              `Expire ratio ${pct(expireRate)}: ${expired} expired / ${terminal} — drives max_expire_rate.\n` +
              `Succeed ratio ${pct(succeedRate)}: ${succeeded} succeeded / ${terminal} — drives max_succeed_ratio.\n` +
              definition;
        if (expireRate == null || succeedRate == null) {
          return (
            <span className="num muted" title={tooltip}>
              —
            </span>
          );
        }
        const ratios = (
          <>
            {pct(expireRate)}
            <span className="muted"> exp</span>
            {" · "}
            {pct(succeedRate)}
            <span className="muted"> ok</span>
          </>
        );
        // Below the min-sample gate the ratios aren't actionable yet. Colour
        // (muted text) alone isn't distinct enough, so add non-colour cues:
        // wrap the ratios in parentheses and append an explicit sample marker
        // (`n <terminal>/<minSamples>`) showing how far off the gate we are.
        if (belowMinSamples) {
          return (
            <span className="num muted" title={tooltip}>
              ({ratios}){" "}
              <span className="datum-audit__gate">
                {terminal}/{Math.ceil(minSamples)}
              </span>
            </span>
          );
        }
        return (
          <span className="num" title={tooltip}>
            {ratios}
          </span>
        );
      },
    }),
    columnHelper.accessor("trained", {
      header: "Trained",
      cell: (info) => <span className="num">{info.getValue()}</span>,
    }),
  ];

  const table = useReactTable({
    data: items,
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
