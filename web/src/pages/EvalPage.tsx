import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { evalProgressQuery, rewardDistributionQuery } from "../api/queries";
import { RewardDistribution } from "../components/RewardDistribution";
import { Mono, Panel } from "../components/ui";

/**
 * Held-out evaluation view: mean reward on the eval split bucketed by the
 * generation of the checkpoint each eval run targeted, plus a per-eval-datum
 * status table. Kept on its own page (rather than mixed into the General tuner
 * page) so the training reward distribution and the held-out eval metric stay
 * clearly separated.
 */
export function EvalPage() {
  const { tunerId } = useParams({ from: "/tuners/$tunerId/eval" });

  const evalDistQ = useQuery(
    rewardDistributionQuery(tunerId, undefined, "eval"),
  );
  const evalProgressQ = useQuery(evalProgressQuery(tunerId));

  const progress = evalProgressQ.data;
  const items = progress?.items ?? [];

  return (
    <div className="page">
      <header className="page__header">
        <h1>Eval</h1>
        <p className="page__subtitle">
          Held-out eval reward, bucketed by the policy generation of the
          checkpoint each eval run scored.
        </p>
      </header>

      <Panel
        title="Held-out eval reward by checkpoint"
            right={
              evalDistQ.isFetching ? (
                <span className="live-dot">● live</span>
              ) : undefined
            }
          >
            {evalDistQ.isError ? (
              <div className="placeholder placeholder--inset placeholder--error">
                Failed to load eval reward: {(evalDistQ.error as Error).message}
              </div>
            ) : !evalDistQ.data ? (
              <div className="placeholder placeholder--inset">
                Loading eval reward…
              </div>
            ) : evalDistQ.data.total === 0 ? (
              <div className="placeholder placeholder--inset">
                No held-out eval runs scored yet — configure{" "}
                <code>eval_datum_ids</code> when creating the tuner, and eval
                scores will appear here once each checkpoint's eval split is
                rewarded.
              </div>
            ) : (
              <RewardDistribution dist={evalDistQ.data} />
            )}
          </Panel>

          <Panel
            title="Eval datums"
            right={
              <span className="muted">
                {progress?.latest_checkpoint_generation != null
                  ? `latest checkpoint gen ${progress.latest_checkpoint_generation}`
                  : "no checkpoint yet"}
              </span>
            }
          >
            {evalProgressQ.isError ? (
              <div className="placeholder placeholder--inset placeholder--error">
                Failed to load eval progress:{" "}
                {(evalProgressQ.error as Error).message}
              </div>
            ) : items.length === 0 ? (
              <div className="placeholder placeholder--inset">
                This tuner has no eval datums. Pass <code>eval_datum_ids</code>{" "}
                at creation to hold out a datum for per-checkpoint scoring.
              </div>
            ) : (
              <div className="table-scroll">
                <table className="table table--dense">
                  <thead>
                    <tr>
                      <th>Datum ID</th>
                      <th className="num">In flight</th>
                      <th className="num">Completed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((it) => (
                      <tr key={it.datum_id}>
                        <td>
                          <Link
                            to="/datums"
                            search={{ tuner: tunerId, datum: it.datum_id }}
                            className="link-plain"
                          >
                            <Mono>{it.datum_id}</Mono>
                          </Link>
                        </td>
                        <td className="num">{it.in_flight}</td>
                        <td className="num">{it.completed}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
    </div>
  );
}
