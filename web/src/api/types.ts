// Friendly aliases over the generated OpenAPI schema (./schema.d.ts).
//
// The schema is generated from the server's OpenAPI document — do NOT edit by
// hand. Regenerate after changing the backend DTOs with:
//
//   uv run poe gen-web-types
//
// This module just maps the `components["schemas"]` names to the flat names the
// rest of the app imports.

import type { components } from "./schema";

type Schemas = components["schemas"];

export type TunerItem = Schemas["TunerItem"];
export type ListTunersResponse = Schemas["ListTunersResponse"];
export type ListDatumsResponse = Schemas["ListDatumsResponse"];
export type Recipe = Schemas["Recipe"];
export type RunProgress = Schemas["RunProgress"];
export type DatumProgress = Schemas["DatumProgress"];
export type NextPick = Schemas["NextPick"];
export type NextPickTier = NextPick["tier"];
export type BatchProgress = Schemas["BatchProgress"];
export type DatumCoverage = Schemas["DatumCoverage"];
export type DatumPool = Schemas["DatumPool"];
export type TrainingProgress = Schemas["TrainingProgress"];
export type EvalDatumProgress = Schemas["EvalDatumProgress"];
export type EvalProgress = Schemas["EvalProgress"];
export type TunerProgress = Schemas["TunerProgress"];
export type GetTunerResponse = Schemas["GetTunerResponse"];
export type RunItem = Schemas["RunItem"];
export type RunStatus = RunItem["status"];
export type ListRunsResponse = Schemas["ListRunsResponse"];
export type GenerationRewardStats = Schemas["GenerationRewardStats"];
export type RewardDistributionData = Schemas["RewardDistributionResponse"];
export type ChatCompletionItem = Schemas["ChatCompletionItem"];
export type RunDetailResponse = Schemas["RunDetailResponse"];
export type ChatCompletionDetailResponse =
  Schemas["ChatCompletionDetailResponse"];
