import type { RunStatus } from "../api/types";
import { Badge } from "./ui";

const STATUS_TONE: Record<
  RunStatus,
  "default" | "good" | "warn" | "danger" | "info"
> = {
  in_flight: "info",
  expired: "warn",
  lost: "danger",
  rewarded: "good",
  trained: "default",
  rejected: "danger",
};

export function RunStatusBadge({ status }: { status: RunStatus }) {
  return <Badge tone={STATUS_TONE[status]}>{status.replace("_", " ")}</Badge>;
}
