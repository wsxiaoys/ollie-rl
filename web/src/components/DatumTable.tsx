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
  // The API returns the trainer's exact corpus order. Preserve it by default;
  // users can still sort any column to inspect the pool another way.
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo(
    () => [
      columnHelper.accessor("next_batch_position", {
        header: "Next batch",
        cell: (info) => {
          const position = info.getValue();
          if (position == null) return <span className="muted">—</span>;
          const ready = info.row.original.consumable >= groupSize;
          return (
            <span
              className={`next-batch-position next-batch-position--${ready ? "ready" : "waiting"}`}
            >
              #{position} · {ready ? "ready" : "waiting"}
            </span>
          );
        },
      }),
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
          const value = info.getValue();
          return (
            <div className="datum-progress">
              <ProgressBar value={value} max={groupSize} tone="good" />
              <span className="datum-progress__label">
                {value}/{groupSize}
              </span>
            </div>
          );
        },
      }),
      columnHelper.accessor("in_flight", {
        header: "In flight",
        cell: (info) => <span className="num">{info.getValue()}</span>,
      }),
      columnHelper.accessor("trained", {
        header: "Trained",
        cell: (info) => <span className="num">{info.getValue()}</span>,
      }),
    ],
    [groupSize, tunerId],
  );

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
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
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
