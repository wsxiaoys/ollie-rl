import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { useEffect } from "react";
import { dataQuery, runsByDatumQuery, tunersQuery } from "../api/queries";
import { RewardDistribution } from "../components/RewardDistribution";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { SearchableSelect } from "../components/SearchableSelect";
import { Mono, Panel } from "../components/ui";

export function DataPage() {
  const { tuner: tunerId, datum: datumId } = useSearch({ from: "/data" });
  const navigate = useNavigate();

  const tunersQ = useQuery(tunersQuery);
  const dataQ = useQuery({
    ...dataQuery(tunerId ?? ""),
    enabled: Boolean(tunerId),
  });
  const runsQ = useQuery({
    ...runsByDatumQuery(tunerId ?? "", datumId ?? ""),
    enabled: Boolean(tunerId) && Boolean(datumId),
  });

  const datumIds = dataQ.data?.datum_ids ?? [];

  // Default the tuner to the first available one so the page is never empty
  // when reached directly (the sidebar normally supplies `?tuner=`).
  useEffect(() => {
    if (!tunerId && tunersQ.data?.tuners && tunersQ.data.tuners.length > 0) {
      navigate({
        to: "/data",
        search: { tuner: tunersQ.data.tuners[0].tuner_id },
      });
    }
  }, [tunerId, tunersQ.data, navigate]);

  // A datum is ALWAYS selected in this view: default to the first datum in the
  // pool as soon as the pool loads.
  useEffect(() => {
    if (tunerId && !datumId && datumIds.length > 0) {
      navigate({ to: "/data", search: { tuner: tunerId, datum: datumIds[0] } });
    }
  }, [tunerId, datumId, datumIds, navigate]);

  const onDatumChange = (value: string | null) => {
    // The datum is always selected; ignore clears.
    if (!value) return;
    navigate({ to: "/data", search: { tuner: tunerId, datum: value } });
  };

  const runs = runsQ.data?.runs ?? [];

  return (
    <div className="page">
      <header className="page__header">
        <h1>Data</h1>
        <p className="page__subtitle">
          Inspect a single datum and every run dispensed for it.
        </p>
      </header>

      <div className="runs-picker">
        <label htmlFor="datum-select">Datum ID</label>
        <SearchableSelect
          id="datum-select"
          value={datumId ?? null}
          options={datumIds}
          onChange={onDatumChange}
          placeholder="Select a datum…"
          searchPlaceholder="Search datum id…"
          clearable={false}
          disabled={!tunerId || datumIds.length === 0}
        />
        {runsQ.isFetching && <span className="live-dot">● live</span>}
      </div>

      {!tunerId && (
        <div className="placeholder">Select a tuner to view its data.</div>
      )}

      {tunerId && dataQ.data && datumIds.length === 0 && (
        <div className="placeholder">This tuner has no registered data.</div>
      )}

      {tunerId && datumId && (
        <>
          <Panel
            title="Reward distribution by generation"
            right={
              runsQ.isFetching ? (
                <span className="live-dot">● live</span>
              ) : undefined
            }
          >
            {runsQ.isError ? (
              <div className="placeholder placeholder--inset placeholder--error">
                Failed to load runs: {(runsQ.error as Error).message}
              </div>
            ) : !runsQ.data ? (
              <div className="placeholder placeholder--inset">
                Loading runs…
              </div>
            ) : (
              <RewardDistribution runs={runs} />
            )}
          </Panel>

          <Panel
            title="Runs"
            right={<span className="muted">{runs.length} runs</span>}
          >
            {runs.length === 0 ? (
              <div className="placeholder placeholder--inset">
                No runs dispensed for this datum yet.
              </div>
            ) : (
              <table className="table table--dense">
                <thead>
                  <tr>
                    <th>Run ID</th>
                    <th>Status</th>
                    <th className="num">Reward</th>
                    <th className="num">Completions</th>
                    <th>Created</th>
                    <th>Expires</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.run_id}>
                      <td>
                        <Link
                          to="/tuners/$tunerId/runs/$runId"
                          params={{ tunerId, runId: r.run_id }}
                          className="link-plain"
                        >
                          <Mono>{r.run_id}</Mono>
                        </Link>
                      </td>
                      <td>
                        <RunStatusBadge status={r.status} />
                      </td>
                      <td className="num">
                        {r.reward === null ? "—" : r.reward.toFixed(3)}
                      </td>
                      <td className="num">{r.completion_count}</td>
                      <td>{new Date(r.created_at).toLocaleString()}</td>
                      <td>{new Date(r.expires_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}
