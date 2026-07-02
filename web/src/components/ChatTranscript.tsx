import { Link } from "@tanstack/react-router";
import { Fragment } from "react";
import type { ChatCompletionItem } from "../api/types";
import { Badge, Mono } from "./ui";
import { PrettyJson } from "./PrettyJson";

type BadgeTone = "default" | "good" | "warn" | "danger" | "info";

/**
 * Map an OpenAI finish reason to a badge tone so the outcome of a turn is
 * legible at a glance: a clean `stop` reads as success, while truncation or
 * filtering stands out as a problem.
 */
const FINISH_REASON_TONE: Record<string, BadgeTone> = {
  stop: "good",
  tool_calls: "info",
  function_call: "info",
  length: "warn",
  content_filter: "danger",
};

type AnyRecord = Record<string, unknown>;

function asRecordArray(value: unknown): AnyRecord[] {
  return Array.isArray(value)
    ? value.filter((v): v is AnyRecord => typeof v === "object" && v !== null)
    : [];
}

/** Flatten OpenAI message content (string or content-part array) to text. */
function renderContent(content: unknown): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object") {
          const p = part as AnyRecord;
          if (typeof p.text === "string") return p.text;
          return JSON.stringify(p, null, 2);
        }
        return String(part);
      })
      .join("\n");
  }
  return JSON.stringify(content, null, 2);
}

/**
 * If `text` is a JSON object or array, parse it and return the object.
 * Otherwise, return null.
 */
function tryParseJson(text: string): unknown | null {
  const trimmed = text.trim();
  if (!/^[[{]/.test(trimmed)) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

/** Recursively sort object keys so semantically-equal JSON compares equal. */
function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortValue);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(value as AnyRecord).sort()) {
      out[k] = sortValue((value as AnyRecord)[k]);
    }
    return out;
  }
  return value;
}

/**
 * Canonicalize a tool-call argument blob so the same call compares equal
 * regardless of whitespace or key ordering. The response copy of an assistant
 * tool call and the copy echoed into the next request often serialize
 * differently (e.g. `{"city": "X"}` vs `{"city":"X"}`); without this the
 * prefix match — and thus trajectory grouping — breaks.
 */
function normalizeArguments(args: unknown): string {
  const raw = typeof args === "string" ? args : JSON.stringify(args ?? "");
  try {
    return JSON.stringify(sortValue(JSON.parse(raw)));
  } catch {
    return raw;
  }
}

function getPromptMessages(completion: ChatCompletionItem): AnyRecord[] {
  const request = completion.request as unknown as AnyRecord;
  return asRecordArray(request.messages);
}

function getResponseMessage(completion: ChatCompletionItem): AnyRecord | null {
  const response = completion.response as unknown as AnyRecord;
  const first = asRecordArray(response.choices)[0];
  if (first && typeof first.message === "object" && first.message !== null) {
    return first.message as AnyRecord;
  }
  return null;
}

function getFinishReason(completion: ChatCompletionItem): string | null {
  const response = completion.response as unknown as AnyRecord;
  const first = asRecordArray(response.choices)[0];
  if (first && typeof first.finish_reason === "string") {
    return first.finish_reason;
  }
  return null;
}

/**
 * Extract the prompt (input) and completion (output) token counts reported in
 * the response `usage`, when present. These are per-turn — each turn is a
 * distinct LLM request whose prompt grows as the trajectory accumulates.
 */
function getUsage(completion: ChatCompletionItem): {
  promptTokens: number;
  completionTokens: number;
  reasoningTokens: number | null;
} | null {
  const response = completion.response as unknown as AnyRecord;
  const usage = response.usage;
  if (!usage || typeof usage !== "object") return null;
  const u = usage as AnyRecord;
  const promptTokens =
    typeof u.prompt_tokens === "number" ? u.prompt_tokens : null;
  const completionTokens =
    typeof u.completion_tokens === "number" ? u.completion_tokens : null;
  if (promptTokens === null && completionTokens === null) return null;
  const details = u.completion_tokens_details as AnyRecord | null | undefined;
  const reasoningTokens =
    details && typeof details.reasoning_tokens === "number"
      ? details.reasoning_tokens
      : null;
  return {
    promptTokens: promptTokens ?? 0,
    completionTokens: completionTokens ?? 0,
    reasoningTokens,
  };
}

/**
 * Canonical, comparison-friendly identity for a message. Two messages are
 * "the same" turn if their role, flattened content, name/tool_call_id, and
 * tool-call name/arguments/id match — robust to incidental fields that differ
 * between a response message and the same message echoed into the next
 * request (e.g. `refusal`, `annotations`).
 */
function messageKey(msg: AnyRecord): string {
  const role = String(msg.role ?? "");
  const content = renderContent(msg.content);
  const name = typeof msg.name === "string" ? msg.name : "";
  const toolCallId =
    typeof msg.tool_call_id === "string" ? msg.tool_call_id : "";
  const toolCalls = asRecordArray(msg.tool_calls).map((tc) => {
    const fn = (tc.function ?? {}) as AnyRecord;
    return {
      id: typeof tc.id === "string" ? tc.id : "",
      name: String(fn.name ?? ""),
      arguments: normalizeArguments(fn.arguments),
    };
  });
  return JSON.stringify({ role, content, name, toolCallId, toolCalls });
}

function isPrefix(prefix: string[], arr: string[]): boolean {
  if (prefix.length > arr.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (prefix[i] !== arr[i]) return false;
  }
  return true;
}

interface ParsedCompletion {
  completion: ChatCompletionItem;
  order: number;
  promptKeys: string[];
  fullKeys: string[];
}

interface TrajectoryGroup {
  items: ParsedCompletion[];
  fullKeys: string[];
}

/**
 * Group chat completions into trajectories via greedy longest-prefix
 * partitioning over their messages — the message-space analogue of the
 * token prefix-tree the Tinker trainer uses to reconstruct trajectories
 * (`src/ollie_rl/trainer/tinker/accumulator.py`).
 *
 * A completion continues a trajectory when that trajectory's running
 * `prompt + response` message sequence is a prefix of the completion's
 * prompt messages. The longest matching prefix wins; otherwise the
 * completion starts a new trajectory.
 */
function buildTrajectories(
  completions: ChatCompletionItem[],
): ChatCompletionItem[][] {
  const parsed: ParsedCompletion[] = completions.map((completion, order) => {
    const promptKeys = getPromptMessages(completion).map(messageKey);
    const responseMessage = getResponseMessage(completion);
    const fullKeys = responseMessage
      ? [...promptKeys, messageKey(responseMessage)]
      : [...promptKeys];
    return { completion, order, promptKeys, fullKeys };
  });

  // Shortest prompts first so a trajectory's earlier turns are seen before
  // the turns that extend them.
  const sorted = [...parsed].sort(
    (a, b) => a.promptKeys.length - b.promptKeys.length || a.order - b.order,
  );

  const groups: TrajectoryGroup[] = [];
  for (const item of sorted) {
    let best: TrajectoryGroup | null = null;
    for (const g of groups) {
      if (isPrefix(g.fullKeys, item.promptKeys)) {
        if (!best || g.fullKeys.length > best.fullKeys.length) best = g;
      }
    }
    if (best) {
      best.items.push(item);
      best.fullKeys = item.fullKeys;
    } else {
      groups.push({ items: [item], fullKeys: item.fullKeys });
    }
  }

  const minOrder = (g: TrajectoryGroup) =>
    Math.min(...g.items.map((i) => i.order));

  return groups
    .sort((a, b) => minOrder(a) - minOrder(b))
    .map((g) =>
      g.items
        .slice()
        .sort((a, b) => a.order - b.order)
        .map((p) => p.completion),
    );
}

/**
 * Whether the model actually invoked a tool anywhere in the trajectory —
 * either an assistant response carrying `tool_calls`, or a follow-up request
 * echoing those tool calls / their `tool` results back into the prompt.
 */
function trajectoryHasToolCall(completions: ChatCompletionItem[]): boolean {
  return completions.some((c) => {
    const responseMessage = getResponseMessage(c);
    if (
      responseMessage &&
      asRecordArray(responseMessage.tool_calls).length > 0
    ) {
      return true;
    }
    return getPromptMessages(c).some(
      (m) => asRecordArray(m.tool_calls).length > 0 || m.role === "tool",
    );
  });
}

/**
 * A trajectory is only "not interesting" (and hidden by default) when it both
 * makes no tool call and consists of a single request turn — typically an
 * auxiliary one-shot request (e.g. title generation) that doesn't reflect the
 * agent's task quality. Any tool call or multi-turn exchange keeps it visible.
 */
function isInterestingTrajectory(completions: ChatCompletionItem[]): boolean {
  const isSingleTurn = completions.length <= 1;
  return trajectoryHasToolCall(completions) || !isSingleTurn;
}

function MessageView({ message }: { message: AnyRecord }) {
  const role = String(message.role ?? "unknown");
  const name = typeof message.name === "string" ? message.name : null;
  const content = renderContent(message.content);
  const toolCalls = asRecordArray(message.tool_calls);

  const roleLabel = (
    <>
      {role}
      {name && <span className="msg__name"> · {name}</span>}
    </>
  );

  const parsedContentJson = content ? tryParseJson(content) : null;

  const body = (
    <>
      {content &&
        (parsedContentJson !== null ? (
          <PrettyJson data={parsedContentJson} expand={1} />
        ) : (
          <div className="msg__content">{content}</div>
        ))}
      {toolCalls.map((tc, i) => {
        const fn = (tc.function ?? {}) as AnyRecord;
        let parsedArgs: unknown = null;
        if (typeof fn.arguments === "string") {
          try {
            parsedArgs = JSON.parse(fn.arguments);
          } catch {
            parsedArgs = fn.arguments;
          }
        } else {
          parsedArgs = fn.arguments;
        }

        return (
          <div key={i} className="msg__tool-call">
            <span className="msg__tool-name">
              🔧 {String(fn.name ?? "tool")}
            </span>
            {parsedArgs !== null && typeof parsedArgs === "object" ? (
              <PrettyJson data={parsedArgs} expand={1} />
            ) : (
              <pre className="msg__tool-args">{String(parsedArgs ?? "")}</pre>
            )}
          </div>
        );
      })}
      {!content && toolCalls.length === 0 && (
        <div className="msg__content msg__content--empty">— empty —</div>
      )}
    </>
  );

  // System prompts are long and rarely change, so collapse them by default.
  if (role === "system") {
    return (
      <details className="msg msg--system">
        <summary className="msg__role msg__role--toggle">{roleLabel}</summary>
        {body}
      </details>
    );
  }

  return (
    <div className={`msg msg--${role}`}>
      <div className="msg__role">{roleLabel}</div>
      {body}
    </div>
  );
}

function TrajectoryCard({
  completions,
  index,
  tunerId,
  runId,
}: {
  completions: ChatCompletionItem[];
  index: number;
  tunerId?: string;
  runId?: string;
}) {
  // Render the trajectory as one continuous conversation: each turn
  // contributes only the messages that are new relative to the previous
  // turn's full sequence (delta prompt) followed by its response — mirroring
  // the accumulator's `extend(delta_prompt, completion)`.
  let prevFullLen = 0;
  const turns = completions.map((completion) => {
    const promptMessages = getPromptMessages(completion);
    const responseMessage = getResponseMessage(completion);
    const deltaPrompt = promptMessages.slice(prevFullLen);
    prevFullLen = promptMessages.length + (responseMessage ? 1 : 0);
    return { completion, deltaPrompt, responseMessage };
  });

  return (
    <section className="trajectory">
      <header className="trajectory__header">
        <span className="trajectory__title">Trajectory #{index + 1}</span>
        <span className="trajectory__meta">
          {completions.length} turn{completions.length === 1 ? "" : "s"}
        </span>
      </header>
      <div className="transcript">
        {turns.map((turn, ti) => (
          <Fragment key={turn.completion.id}>
            {turn.deltaPrompt.map((m, mi) => (
              <MessageView key={`p-${ti}-${mi}`} message={m} />
            ))}
            <div className="turn-divider">
              <span className="turn-divider__label">turn {ti + 1}</span>
              {(() => {
                const finishReason = getFinishReason(turn.completion);
                const title =
                  finishReason === "content_filter"
                    ? "Malformed output (e.g. bad function call, bad reasoning block) is also adapted into this stop reason"
                    : undefined;
                return (
                  finishReason && (
                    <Badge
                      tone={FINISH_REASON_TONE[finishReason] ?? "default"}
                      title={title}
                    >
                      {finishReason}
                    </Badge>
                  )
                );
              })()}
              {(() => {
                const usage = getUsage(turn.completion);
                return (
                  usage && (
                    <span
                      className="turn-divider__meta turn-divider__meta--tokens"
                      title={
                        `${usage.promptTokens} input / ${usage.completionTokens} output tokens` +
                        (usage.reasoningTokens !== null
                          ? ` (${usage.reasoningTokens} reasoning)`
                          : "")
                      }
                    >
                      <span className="turn-divider__meta-label">tokens</span>
                      <Mono>
                        {usage.promptTokens.toLocaleString()} in ·{" "}
                        {usage.completionTokens.toLocaleString()} out
                        {usage.reasoningTokens !== null && (
                          <> · {usage.reasoningTokens.toLocaleString()} reason</>
                        )}
                      </Mono>
                    </span>
                  )
                );
              })()}
              <span className="turn-divider__spacer" />
              <span className="turn-divider__meta">
                <span className="turn-divider__meta-label">id</span>
                {tunerId && runId ? (
                  <Link
                    to="/tuners/$tunerId/runs/$runId/completions/$completionId"
                    params={{
                      tunerId,
                      runId,
                      completionId: turn.completion.id,
                    }}
                    className="link"
                  >
                    <Mono>{turn.completion.id}</Mono>
                  </Link>
                ) : (
                  <Mono>{turn.completion.id}</Mono>
                )}
              </span>
              <span className="turn-divider__meta">
                <span className="turn-divider__meta-label">gen</span>
                {turn.completion.policy_generation}
              </span>
              <span className="turn-divider__meta">
                {new Date(turn.completion.created_at).toLocaleString()}
              </span>
            </div>
            {turn.responseMessage ? (
              <MessageView message={turn.responseMessage} />
            ) : (
              <div className="placeholder placeholder--inset">
                No response message recorded.
              </div>
            )}
          </Fragment>
        ))}
      </div>
    </section>
  );
}

export function ChatTranscript({
  completions,
  showToollessTrajectories = false,
  tunerId,
  runId,
}: {
  completions: ChatCompletionItem[];
  showToollessTrajectories?: boolean;
  tunerId?: string;
  runId?: string;
}) {
  if (completions.length === 0) {
    return (
      <div className="placeholder placeholder--inset">
        No chat completions recorded for this run yet.
      </div>
    );
  }

  const trajectories = buildTrajectories(completions);
  const visible = showToollessTrajectories
    ? trajectories
    : trajectories.filter(isInterestingTrajectory);
  const hiddenCount = trajectories.length - visible.length;

  return (
    <div className="transcript-list">
      {hiddenCount > 0 && (
        <div className="transcript-note">
          {hiddenCount} tool-less trajector{hiddenCount === 1 ? "y" : "ies"}{" "}
          hidden
        </div>
      )}
      {visible.length === 0 ? (
        <div className="placeholder placeholder--inset">
          All trajectories are tool-less — enable “Show tool-less trajectories”
          to view them.
        </div>
      ) : (
        visible.map((traj, i) => (
          <TrajectoryCard
            key={traj[0].id}
            completions={traj}
            index={i}
            tunerId={tunerId}
            runId={runId}
          />
        ))
      )}
    </div>
  );
}
