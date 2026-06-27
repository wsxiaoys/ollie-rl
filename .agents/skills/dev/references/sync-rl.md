# How to Interact with the Ollie RL API Server (Sync RL)

This reference describes how a **synchronous RL** client drives the Ollie
RL api server over its public HTTP API. By "sync RL" we mean the
canonical GRPO loop where sampling pauses while training is in flight,
and the server dispenses run assignments to coordinate multiple parallel workers.

```mermaid
sequenceDiagram
    participant C as Sync RL Client / Worker
    participant API as Ollie RL API
    participant DB as DB

    C->>API: POST /tuners { recipe, datum_ids: [...] }
    API-->>C: { tuner_id }

    loop each training step
        loop fan out N samplers (parallel)
            C->>API: POST /tuners/{id}/runs
            alt queue open
                API->>API: dispense_run(is_training=False, datum_pool)
                API->>DB: INSERT runs
                API-->>C: 200 { run_id, datum_id, expires_at }
                C->>API: POST /openai/v1/chat/completions { x_run_id, ... }
                API->>DB: INSERT chat_completions (policy_generation)
                API-->>C: ChatCompletion
                C->>API: PUT /tuners/{id}/runs/{run_id}/reward { reward }
                API->>DB: UPDATE runs SET reward=…
            else barrier closed
                API-->>C: 204 + Retry-After
                Note over C: backoff and retry
            end
        end
    end
```

A **run** is the unit of reward / advantage. A single run may internally
contain multiple trajectories (e.g. multi-step or agent-with-sub-agent setups);
they all share the same `run_id`, reward, and advantage.

## API Surface the Client Talks To

| Endpoint                                       | Purpose                                   |
|------------------------------------------------|-------------------------------------------|
| `POST /tuners`                                 | Create a tuner with a registered datum pool. |
| `POST /tuners/{tuner_id}/runs`                 | Request a new run assignment.             |
| `POST /openai/v1/chat/completions`             | Sample one LLM response inside a `run_id`.|
| `PUT /tuners/{tuner_id}/runs/{run_id}/reward`  | Submit the scalar reward for a `run_id`.  |
| `GET /tuners/{tuner_id}`                       | Get the status of a tuner (observability). |

Training is applied implicitly by the server as rewards arrive; the client does not need to trigger it explicitly.

### Required headers on `/openai/v1/chat/completions`

| Header        | Required | Meaning                                                     |
|---------------|----------|-------------------------------------------------------------|
| `X-Tuner-Id`  | yes      | Which tuner / policy to sample from.                        |
| `X-Run-Id`    | yes\*    | The run this completion belongs to.                         |

\* `X-Run-Id` should be sent for **any request whose output affects the final result** the agent is being scored on (i.e. completions that participate in solving the task and will be used as training examples). Auxiliary requests that do not affect the result — for example generating a chat title, summarizing logs, or other side-channel calls that are not part of the task context — should **omit** the header so they are not recorded as training examples. Note that the server automatically maps the `run_id` to its assigned `datum_id` to prevent client-side tampering.

## One Training Step, Visualized

A single sync-RL step has three phases visible to the client.

### Phase 0 — bootstrap (once per training job)

`POST /tuners` with the recipe payload and `datum_ids` (non-empty list) to create a tuner.
The server returns a `tuner_id`. Persist it somewhere durable — it is the only handle to the policy on the server.

### Phase 1 — request run assignments

Workers request work by calling `POST /tuners/{tuner_id}/runs` with an empty body:

- **200 OK**: Returns `{ run_id, datum_id, expires_at }`. The worker should execute the run.
- **204 No Content**: The barrier is closed (e.g., training is in flight). The response will include a `Retry-After` header (usually `1` second). The worker should back off and retry.

### Phase 2 — execute run and submit reward

For every dispensed `(datum_id, run_id)`, the client drives an agent run that may issue **multiple** chat completion calls (multi-turn dialogue, tool use, sub-agent calls, etc.), all sharing the same `run_id`. Once the run terminates, the client submits one scalar reward for it via `PUT /tuners/{tuner_id}/runs/{run_id}/reward`.

```mermaid
sequenceDiagram
    participant C as Client (Worker)
    participant API as Ollie RL API

    C->>API: POST /tuners/{id}/runs
    API-->>C: 200 { run_id, datum_id, expires_at }
    loop one or more turns per run
        C->>API: POST /openai/v1/chat/completions<br/>X-Tuner-Id, X-Run-Id
        API-->>C: ChatCompletion
        Note over C: agent acts (tools, sub-agents, next turn…)
    end
    Note over C: client scores the run
    C->>API: PUT /tuners/{tuner_id}/runs/{run_id}/reward<br/>{ reward: 0.75 }
    API-->>C: 200 OK
```

### Phase 3 — loop

The client simply continues the loop, requesting the next run assignment. The server manages group sizes, batching, and training barriers internally.

## Things a Sync-RL Client Must Get Right

- **Never reuse a `run_id`.** The server allocates the `run_id` dynamically; always use the `run_id` dispensed by the server.
- **Send `X-Run-Id` on result-affecting completions.** Without it, the completion is not recorded and the run cannot contribute to training. Conversely, omit it on auxiliary calls so they are not picked up as training examples.
- **Submit rewards before the run expires.** Every run is leased with an expiration deadline (`expires_at`, default `5 minutes`). If a reward is posted after the lease expires, the server returns `409 Conflict` because the dispenser may have already re-issued that datum to another worker.
- **Rewards are write-once.** Once a reward has been submitted for a `run_id`, it cannot be changed. Subsequent `PUT /reward` calls on the same `run_id` return `409 Conflict`.
- **Pace yourself.** The server does not limit concurrent completions or rewards. A sync-RL driver should bound its own fan-out so the server is not overwhelmed by a flood of HTTP work.
