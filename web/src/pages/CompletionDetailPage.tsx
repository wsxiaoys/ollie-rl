import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import type { ChatCompletionItem } from "../api/types";
import { completionQuery } from "../api/queries";
import { ChatTranscript } from "../components/ChatTranscript";
import { Mono, Panel, StatCard } from "../components/ui";

export function CompletionDetailPage() {
  const { tunerId, runId, completionId } = useParams({
    from: "/tuners/$tunerId/runs/$runId/completions/$completionId",
  });
  const { data, isLoading, isError, error } = useQuery(
    completionQuery(tunerId, runId, completionId),
  );

  if (isLoading) {
    return <div className="placeholder">Loading completion…</div>;
  }
  if (isError) {
    return (
      <div className="placeholder placeholder--error">
        Failed to load completion: {(error as Error).message}
      </div>
    );
  }
  if (!data) return null;

  // Reuse the run transcript renderer for the single completion. The detail
  // response is a superset of `ChatCompletionItem`, so it renders the prompt
  // messages and the response turn exactly as they appear in the run view.
  const item: ChatCompletionItem = {
    id: data.id,
    policy_generation: data.policy_generation,
    created_at: data.created_at,
    request: data.request,
    response: data.response,
  };

  return (
    <div className="page">
      <header className="page__header">
        <Link
          to="/tuners/$tunerId/runs/$runId"
          params={{ tunerId, runId }}
          className="link link--back"
        >
          ← Run
        </Link>
        <h1>Completion detail</h1>
        <div className="detail-header__meta">
          <Mono>{data.id}</Mono>
        </div>
      </header>

      <div className="kpi-strip kpi-strip--runs">
        <StatCard
          label="run"
          value={
            <Link
              to="/tuners/$tunerId/runs/$runId"
              params={{ tunerId, runId }}
              className="link"
            >
              <Mono>{data.run_id}</Mono>
            </Link>
          }
        />
        <StatCard label="datum" value={<Mono>{data.datum_id}</Mono>} />
        <StatCard label="policy gen" value={data.policy_generation} />
        <StatCard
          label="tokens"
          value={data.tokens ? data.tokens.length : "—"}
          tone={data.tokens ? "default" : "muted"}
        />
        <StatCard
          label="logprobs"
          value={data.logprobs ? data.logprobs.length : "—"}
          tone={data.logprobs ? "default" : "muted"}
        />
      </div>

      <Panel title="Transcript">
        <ChatTranscript completions={[item]} showToollessTrajectories />
      </Panel>

      <Panel title="Raw request">
        <pre className="msg__json">
          {JSON.stringify(data.request, null, 2)}
        </pre>
      </Panel>

      <Panel title="Raw response">
        <pre className="msg__json">
          {JSON.stringify(data.response, null, 2)}
        </pre>
      </Panel>
    </div>
  );
}
