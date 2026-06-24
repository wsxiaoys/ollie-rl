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
    candidates_token_count: int
    total_token_count: int


class GenerateContentTuningScopeResponse(BaseModelConfig):
    candidates: Dict[str, Candidate] = Field(default_factory=dict)
    tuning_candidates: List[TuningCandidate] = Field(default_factory=list)
    usage_metadata: Optional[UsageMetadata] = None
    train_step_id: Optional[str] = None


# --- TrainStep Models ---


class ReinforcementTuningTrainingData(BaseModelConfig):
    candidate_id: str
    advantage: float


class ReinforcementTuningTrainingDataBatch(BaseModelConfig):
    examples: List[ReinforcementTuningTrainingData]


class TrainStepRequest(BaseModelConfig):
    reinforcement_tuning_training_data_batch: ReinforcementTuningTrainingDataBatch


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


class TrainStepMetric(BaseModelConfig):
    # Flexible container for TrainStepMetric fields
    model_config = ConfigDict(extra="allow")


class TrainStepResponse(BaseModelConfig):
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
