from typing import Any, Dict, List, Optional, TypeVar, Type
from pydantic import BaseModel, ConfigDict, Field, alias_generators
from google.genai.types import (
    Candidate,
    Content,
    Tool,
)

T = TypeVar("T", bound=BaseModel)


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
        extra="forbid",
        # This allows us to use arbitrary types in the model. E.g. PIL.Image.
        arbitrary_types_allowed=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        ignored_types=(TypeVar,),
    )


# --- Common / LRO Models ---


class Status(BaseModelConfig):
    code: int
    message: str
    details: Optional[List[Dict[str, Any]]] = None


class GenericMetadata(BaseModelConfig):
    create_time: Optional[str] = None
    update_time: Optional[str] = None


class GenerateContentTuningScopeOperationMetadata(BaseModelConfig):
    generic_metadata: Optional[GenericMetadata] = None


class TrainStepOperationMetadata(BaseModelConfig):
    generic_metadata: Optional[GenericMetadata] = None


# --- TuningJob Models ---


class MultiStepReinforcementTuningHyperParameters(BaseModelConfig):
    adapter_size: Optional[str] = None
    checkpoint_interval: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None


class MultiStepReinforcementTuningSpec(BaseModelConfig):
    hyper_parameters: Optional[MultiStepReinforcementTuningHyperParameters] = None


class CreateTuningJobRequest(BaseModelConfig):
    tuned_model_display_name: str
    description: Optional[str] = None
    base_model: str
    multi_step_reinforcement_tuning_spec: Optional[MultiStepReinforcementTuningSpec] = (
        None
    )


class TuningJobMetadata(BaseModelConfig):
    # Flexible container for TuningJobMetadata fields
    model_config = ConfigDict(extra="allow")


class TuningJob(BaseModelConfig):
    # Response-side model: Vertex returns many fields that are not captured here
    # (e.g. start_time, experiment, tuned_model). Allow extras so the client
    # is resilient to API evolution. Fields we actively read remain strict.
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
        extra="allow",
        arbitrary_types_allowed=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        ignored_types=(TypeVar,),
    )

    name: str
    tuned_model_display_name: str
    description: Optional[str] = None
    base_model: str
    state: Optional[str] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    multi_step_reinforcement_tuning_spec: Optional[MultiStepReinforcementTuningSpec] = (
        None
    )
    metadata: Optional[TuningJobMetadata] = None
    error: Optional[Status] = None


# --- GenerateContentTuningScope Models ---


class GenerationConfig(BaseModelConfig):
    max_output_tokens: Optional[int] = None


class ContentGenerationParameters(BaseModelConfig):
    contents: List[Content]
    generation_config: Optional[GenerationConfig] = None
    system_instruction: Optional[Content] = None
    tools: Optional[List[Tool]] = None


class GenerateContentTuningScopeRequest(BaseModelConfig):
    content_generation_parameters: ContentGenerationParameters


class TuningCandidate(BaseModelConfig):
    candidate_id: str
    candidate: Candidate
    generation_token_count: Optional[int] = None
    thoughts_token_count: Optional[int] = None


class UsageMetadata(BaseModelConfig):
    prompt_token_count: int
    # Optional: absent when generation was cut off before a visible candidate
    # was produced (e.g. all tokens were consumed by reasoning/"thinking").
    candidates_token_count: Optional[int] = None
    total_token_count: int
    # Optional reasoning/thinking token count returned by Gemini "thinking"
    # variants. Not present on every response.
    thoughts_token_count: Optional[int] = None


class GenerateContentTuningScopeResponse(BaseModelConfig):
    # Vertex returns the standard protobuf Any-wrapper `@type` discriminator
    # (plus potentially other future fields). Allow extras here so we don't
    # have to chase response-shape drift one field at a time.
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
        extra="allow",
        arbitrary_types_allowed=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        ignored_types=(TypeVar,),
    )
    candidates: Dict[str, Candidate] = Field(default_factory=dict)
    tuning_candidates: List[TuningCandidate] = Field(default_factory=list)
    usage_metadata: Optional[UsageMetadata] = None
    train_step_id: str


# --- TrainStep Models ---


class ReinforcementTuningTrainingData(BaseModelConfig):
    candidate_id: str
    advantage: float


class ReinforcementTuningTrainingDataBatch(BaseModelConfig):
    examples: List[ReinforcementTuningTrainingData]


class TrainStepRequest(BaseModelConfig):
    reinforcement_tuning_training_data_batch: ReinforcementTuningTrainingDataBatch
    # When true, the backend applies the gradient step but skips syncing the
    # updated weights to the serving/sampler path. Used to promote a sampler
    # only every N steps (see Recipe.sampler_promotion_every) rather than on
    # every train step. Serialized as `skipWeightSync`.
    skip_weight_sync: Optional[bool] = None


class UsedCandidate(BaseModelConfig):
    candidate_id: str


class RejectedCandidate(BaseModelConfig):
    candidate_id: str
    reason: Optional[str] = None


class CandidateAudit(BaseModelConfig):
    used_candidates: Optional[List[UsedCandidate]] = None
    rejected_candidates: Optional[List[RejectedCandidate]] = None


class TunedModelCheckpoint(BaseModelConfig):
    # Flexible container for TunedModelCheckpoint fields
    model_config = ConfigDict(extra="allow")

    # The ID of the checkpoint.
    checkpoint_id: str
    # The epoch of the checkpoint.
    epoch: int
    # The training step this checkpoint was produced at. Used as the
    # checkpoint's `policy_generation`. Vertex serializes int64 as a string,
    # so accept both and coerce to int.
    step: int
    # Endpoint resource name that the checkpoint is deployed to. Ollie persists
    # this as the checkpoint ref used for frozen/eval sampling.
    endpoint: str


class TrainStepMetric(BaseModelConfig):
    # Flexible container for TrainStepMetric fields
    model_config = ConfigDict(extra="allow")


class TrainStepResponse(BaseModelConfig):
    # Vertex wraps the response in a protobuf Any with a `@type` discriminator
    # (plus potentially other future fields). Allow extras so parsing doesn't
    # break on response-shape drift. Mirrors the policy used by
    # `GenerateContentTuningScopeResponse`.
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
        extra="allow",
        arbitrary_types_allowed=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        ignored_types=(TypeVar,),
    )
    completed_train_step_id: Optional[str] = None
    candidate_audit: Optional[CandidateAudit] = None
    tuned_model_checkpoint: Optional[TunedModelCheckpoint] = None
    train_step_metric: Optional[TrainStepMetric] = None


# --- General Operation Model ---


class Operation(BaseModelConfig):
    name: str
    metadata: Optional[Dict[str, Any]] = None
    done: Optional[bool] = False
    error: Optional[Status] = None
    response: Optional[Dict[str, Any]] = None

    def get_response_as(self, response_type: Type[T]) -> Optional[T]:
        """Helper to parse the dynamic response field into a specific model type."""
        if not self.response:
            return None
        return response_type.model_validate(self.response)
