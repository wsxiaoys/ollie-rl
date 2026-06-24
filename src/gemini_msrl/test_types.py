import unittest

from gemini_msrl.types import (
    ContentGenerationParameters,
    CreateTuningJobRequest,
    GenerateContentTuningScopeRequest,
    GenerateContentTuningScopeResponse,
    GenerationConfig,
    TrainStepRequest,
    TuningJob,
)

from google.genai.types import (
    Content,
    Part,
)


class TestGeminiMsrlTypes(unittest.TestCase):
    def test_generate_content_tuning_scope_response_parsing(self):
        # Sample response payload based on api-spec.md
        payload = {
            "candidates": {
                "774a815e-0a72-4564-9ee3-94e8028fd750_0": {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": "```python\ndef two_sum(nums, target):\n    # implementation\n    ...\n```",
                                "thoughtSignature": "YWJjZA==",
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            },
            "tuningCandidates": [
                {
                    "candidateId": "774a815e-0a72-4564-9ee3-94e8028fd750_0",
                    "candidate": {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": "```python\ndef two_sum(nums, target):\n    # implementation\n    ...\n```",
                                    "thoughtSignature": "YWJjZA==",
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    },
                    "generationTokenCount": 120,
                    "thoughtsTokenCount": 45,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 270,
                "totalTokenCount": 285,
            },
            "trainStepId": "1",
        }

        # Parse the response
        response = GenerateContentTuningScopeResponse.model_validate(payload)

        # Assertions to verify correct parsing with standard google.genai types
        self.assertIn("774a815e-0a72-4564-9ee3-94e8028fd750_0", response.candidates)
        candidate = response.candidates["774a815e-0a72-4564-9ee3-94e8028fd750_0"]

        # Verify content and part are correctly parsed
        self.assertIsNotNone(candidate.content)
        assert candidate.content is not None
        self.assertEqual(candidate.content.role, "model")
        self.assertIsNotNone(candidate.content.parts)
        assert candidate.content.parts is not None
        self.assertEqual(len(candidate.content.parts), 1)
        self.assertEqual(
            candidate.content.parts[0].text,
            "```python\ndef two_sum(nums, target):\n    # implementation\n    ...\n```",
        )

        # Verify thought_signature parsed as bytes (standard google.genai.types.Part thought_signature is bytes)
        self.assertTrue(isinstance(candidate.content.parts[0].thought_signature, bytes))

        # Verify finish_reason is standard CaseInSensitiveEnum and compares equal to "STOP"
        self.assertEqual(candidate.finish_reason, "STOP")

        # Verify tuning_candidates
        self.assertEqual(len(response.tuning_candidates), 1)
        tc = response.tuning_candidates[0]
        self.assertEqual(tc.candidate_id, "774a815e-0a72-4564-9ee3-94e8028fd750_0")
        self.assertEqual(tc.generation_token_count, 120)
        self.assertEqual(tc.thoughts_token_count, 45)

        # Verify usage_metadata
        self.assertIsNotNone(response.usage_metadata)
        assert response.usage_metadata is not None
        self.assertEqual(response.usage_metadata.prompt_token_count, 15)
        self.assertEqual(response.usage_metadata.candidates_token_count, 270)
        self.assertEqual(response.usage_metadata.total_token_count, 285)

        # Verify serialization back to dict with alias (camelCase)
        dumped = response.model_dump(by_alias=True, exclude_none=True)
        self.assertIn("tuningCandidates", dumped)
        self.assertIn("usageMetadata", dumped)
        self.assertEqual(dumped["usageMetadata"]["candidatesTokenCount"], 270)

    def test_request_serialization(self):
        # Build request using standard google.genai types
        req = GenerateContentTuningScopeRequest(
            content_generation_parameters=ContentGenerationParameters(
                contents=[Content(role="user", parts=[Part(text="Hello world")])],
                generation_config=GenerationConfig(
                    max_output_tokens=100,
                ),
                system_instruction=Content(
                    role="user", parts=[Part(text="Be concise")]
                ),
                tools=[],
            )
        )

        dumped = req.model_dump(by_alias=True, exclude_none=True)
        self.assertIn("contentGenerationParameters", dumped)
        params = dumped["contentGenerationParameters"]
        self.assertIn("contents", params)
        self.assertIn("generationConfig", params)
        self.assertIn("systemInstruction", params)
        self.assertIn("tools", params)
        self.assertEqual(params["generationConfig"]["maxOutputTokens"], 100)
        self.assertEqual(params["systemInstruction"]["parts"][0]["text"], "Be concise")

    def test_create_tuning_job_request_parsing_from_curl(self):
        # Exact payload from CreateTuningJob curl example in api-spec.md
        payload = {
            "tuned_model_display_name": "my-msrl-model",
            "description": "Multi-Step Reinforcement Learning Tuning Job",
            "base_model": "gemini-3.5-flash",
            "multi_step_reinforcement_tuning_spec": {
                "hyper_parameters": {
                    "adapter_size": "ADAPTER_SIZE_SIXTEEN",
                    "checkpoint_interval": 10,
                }
            },
        }

        req = CreateTuningJobRequest.model_validate(payload)
        self.assertEqual(req.tuned_model_display_name, "my-msrl-model")
        self.assertEqual(
            req.description, "Multi-Step Reinforcement Learning Tuning Job"
        )
        self.assertEqual(req.base_model, "gemini-3.5-flash")

        self.assertIsNotNone(req.multi_step_reinforcement_tuning_spec)
        assert req.multi_step_reinforcement_tuning_spec is not None
        self.assertIsNotNone(req.multi_step_reinforcement_tuning_spec.hyper_parameters)
        assert req.multi_step_reinforcement_tuning_spec.hyper_parameters is not None
        self.assertEqual(
            req.multi_step_reinforcement_tuning_spec.hyper_parameters.adapter_size,
            "ADAPTER_SIZE_SIXTEEN",
        )
        self.assertEqual(
            req.multi_step_reinforcement_tuning_spec.hyper_parameters.checkpoint_interval,
            10,
        )

        # Verify serialization to camelCase
        dumped = req.model_dump(by_alias=True, exclude_none=True)
        self.assertIn("tunedModelDisplayName", dumped)
        self.assertIn("multiStepReinforcementTuningSpec", dumped)
        self.assertEqual(dumped["tunedModelDisplayName"], "my-msrl-model")

    def test_generate_content_tuning_scope_request_parsing_from_curl(self):
        # Exact payload from GenerateContentTuningScope curl example in api-spec.md
        payload = {
            "content_generation_parameters": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Write a python script to solve two sum."}],
                    }
                ],
                "generation_config": {"maxOutputTokens": 8192},
            }
        }

        req = GenerateContentTuningScopeRequest.model_validate(payload)
        params = req.content_generation_parameters
        self.assertEqual(len(params.contents), 1)
        self.assertEqual(params.contents[0].role, "user")
        self.assertIsNotNone(params.contents[0].parts)
        assert params.contents[0].parts is not None
        self.assertEqual(len(params.contents[0].parts), 1)
        self.assertEqual(
            params.contents[0].parts[0].text, "Write a python script to solve two sum."
        )

        self.assertIsNotNone(params.generation_config)
        assert params.generation_config is not None
        self.assertEqual(params.generation_config.max_output_tokens, 8192)

        # Verify serialization to camelCase
        dumped = req.model_dump(by_alias=True, exclude_none=True)
        self.assertIn("contentGenerationParameters", dumped)
        self.assertIn(
            "maxOutputTokens", dumped["contentGenerationParameters"]["generationConfig"]
        )

    def test_train_step_request_parsing_from_curl(self):
        # Exact payload from TrainStep curl example in api-spec.md
        payload = {
            "reinforcement_tuning_training_data_batch": {
                "examples": [
                    {
                        "candidate_id": "774a815e-0a72-4564-9ee3-94e8028fd750_0",
                        "advantage": 0.75,
                    },
                    {
                        "candidate_id": "774a815e-0a72-4564-9ee3-94e8028fd750_1",
                        "advantage": -0.25,
                    },
                ]
            }
        }

        req = TrainStepRequest.model_validate(payload)
        batch = req.reinforcement_tuning_training_data_batch
        self.assertEqual(len(batch.examples), 2)
        self.assertEqual(
            batch.examples[0].candidate_id, "774a815e-0a72-4564-9ee3-94e8028fd750_0"
        )
        self.assertEqual(batch.examples[0].advantage, 0.75)
        self.assertEqual(
            batch.examples[1].candidate_id, "774a815e-0a72-4564-9ee3-94e8028fd750_1"
        )
        self.assertEqual(batch.examples[1].advantage, -0.25)

        # Verify serialization to camelCase
        dumped = req.model_dump(by_alias=True, exclude_none=True)
        self.assertIn("reinforcementTuningTrainingDataBatch", dumped)
        examples = dumped["reinforcementTuningTrainingDataBatch"]["examples"]
        self.assertEqual(
            examples[0]["candidateId"], "774a815e-0a72-4564-9ee3-94e8028fd750_0"
        )

    def test_tuning_job_parsing_with_metadata(self):
        payload = {
            "name": "projects/123/locations/us-central1/tuningJobs/456",
            "tuned_model_display_name": "my-msrl-model",
            "base_model": "gemini-3.5-flash",
            "state": "JOB_STATE_RUNNING",
            "metadata": {"custom_field": "some-value", "another_field": 123},
        }

        job = TuningJob.model_validate(payload)
        self.assertEqual(job.name, "projects/123/locations/us-central1/tuningJobs/456")
        self.assertEqual(job.tuned_model_display_name, "my-msrl-model")
        self.assertEqual(job.base_model, "gemini-3.5-flash")
        self.assertEqual(job.state, "JOB_STATE_RUNNING")

        self.assertIsNotNone(job.metadata)
        assert job.metadata is not None
        # Verify extra fields are allowed and accessible via model_extra
        self.assertIsNotNone(job.metadata.model_extra)
        assert job.metadata.model_extra is not None
        self.assertEqual(job.metadata.model_extra.get("custom_field"), "some-value")
        self.assertEqual(job.metadata.model_extra.get("another_field"), 123)

        # Verify serialization to camelCase
        dumped = job.model_dump(by_alias=True, exclude_none=True)
        self.assertEqual(dumped["metadata"]["custom_field"], "some-value")
        self.assertEqual(dumped["metadata"]["another_field"], 123)
