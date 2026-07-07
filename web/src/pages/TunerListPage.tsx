import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { tunersQuery } from "../api/queries";
import { Badge, Mono } from "../components/ui";

export function TunerListPage() {
  const { data, isLoading, isError, error } = useQuery(tunersQuery);

  return (
    <div className="page">
      <header className="page__header">
        <h1>Tuners</h1>
        <p className="page__subtitle">
          Live training jobs. Click a tuner to inspect its progress.
        </p>
      </header>

      {isLoading && <div className="placeholder">Loading tuners…</div>}
      {isError && (
        <div className="placeholder placeholder--error">
          Failed to load tuners: {(error as Error).message}
        </div>
      )}

      {data && data.tuners.length === 0 && (
        <div className="placeholder">
          No tuners yet. Create one with <Mono>POST /tuners</Mono>.
        </div>
      )}

      {data && data.tuners.length > 0 && (
        <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Tuner ID</th>
              <th>Trainer</th>
              <th className="num">Policy gen</th>
            </tr>
          </thead>
          <tbody>
            {data.tuners.map((t) => (
              <tr key={t.tuner_id}>
                <td>
                  <Link
                    to="/tuners/$tunerId"
                    params={{ tunerId: t.tuner_id }}
                    className="link-plain"
                  >
                    {t.name}
                  </Link>
                </td>
                <td>
                  <Mono>{t.tuner_id}</Mono>
                </td>
                <td>
                  <Badge tone="info">{t.trainer}</Badge>
                </td>
                <td className="num">{t.policy_generation}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </div>
  );
}
