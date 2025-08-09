"""
Multi-LLM Orchestration Service with Intelligent Fallback and Response Merging
"""

import re
from enum import Enum
from typing import List, Dict, Any
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

from core.logger import get_logger
from services.local_llm_service import LocalLLMService
from services.remote_llm_service import RemoteLLMService
from services.response_merger import ResponseMerger

logger = get_logger(__name__)


class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    PROJECT_CREATION = "project_creation"
    MOBILE_APP = "mobile_app"
    FILE_OPERATION = "file_operation"
    CODE_EXECUTION = "code_execution"
    BUG_FIXING = "bug_fixing"
    TESTING = "testing"
    API_CREATION = "api_creation"
    DATABASE_DESIGN = "database_design"
    DEPLOYMENT = "deployment"
    IMAGE_CREATION = "image_creation"
    VIDEO_CREATION = "video_creation"
    UI_DESIGN = "ui_design"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    PRESENTATION = "presentation"
    CONTENT_WRITING = "content_writing"
    MUSIC_GENERATION = "music_generation"
    GAME_DEVELOPMENT = "game_development"
    AUTOMATION = "automation"
    GENERAL_CHAT = "general_chat"  # fallback


@dataclass
class LLMResponse:
    """LLM response with metadata"""
    text: str
    model: str
    provider: str
    response_time: float
    quality_score: float
    tokens_used: int
    metadata: Dict[str, Any]

class LLMOrchestrator:
    """Orchestrates multiple LLMs with intelligent fallback and response merging"""

    def __init__(self, settings):
        self.settings = settings
        self.local_llm = LocalLLMService(settings)
        self.remote_llm = RemoteLLMService(settings)
        self.response_merger = ResponseMerger(settings)
        
        # Configure providers with priorities
        self.llm_priority = self._configure_providers()
        
        self.llm_priority = [
            # Local LLMs (via Ollama)
            {"provider": "local", "model": "ollama", "weight": 1.0},
            # You can add more local providers here if needed


            # Remote LLMs (fallback)
            {"provider": "remote", "model": "together", "weight": 0.7},
            {"provider": "remote", "model": "openrouter", "weight": 0.6},
            {"provider": "remote", "model": "groq", "weight": 0.5},
            {"provider": "remote", "model": "huggingface", "weight": 0.4},
            {"provider": "remote", "model": "openai", "weight": 0.3},
            {"provider": "remote", "model": "anthropic", "weight": 0.2},

            
            # Free remote LLMs
            {"provider": "remote", "model": "groq-llama3", "weight": 0.8},
            {"provider": "remote", "model": "deepseek-coder", "weight": 0.7},
            {"provider": "remote", "model": "openrouter-mistral", "weight": 0.6},

            # Paid remote LLMs (fallback)
            {"provider": "remote", "model": "gpt-4", "weight": 0.5},
            {"provider": "remote", "model": "gpt-3.5-turbo", "weight": 0.4},
            {"provider": "remote", "model": "claude-3-haiku", "weight": 0.3}
        ]

        # Task-specific model preferences
        self.task_preferences = {
            'code_generation': ['llama2', 'codellama', 'deepseek-coder'],
            'code_fix': ['llama2', 'codellama', 'gpt-3.5-turbo'],
            'code_review': ['gpt-4', 'claude-3', 'llama3'],
            'research': ['llama3', 'llama2', 'dolphin'],
            'creative': ['mythomax', 'dolphin', 'openchat'],
            'analytical': ['llama3', 'llama2', 'gpt-4'],
            'conversational': ['llama3', 'llama2', 'claude-3']
        }
        # Performance tracking
        self.performance_history = {}
        self.available_models = {}
        self.failure_counts = {}

    def _configure_providers(self) -> List[Dict[str, Any]]:
        return [
            # Local LLMs (use provider keys here!)
           
        ]
    
    async def initialize(self):
        """Initialize LLM services"""
        logger.info("Initializing LLM orchestrator...")
        
        # Initialize LLM services
        await self.local_llm.initialize()
        await self.remote_llm.initialize()
        
        # Check available models
        await self._check_available_models()
        
        logger.info("LLM orchestrator initialized successfully")
        logger.info(f"Available models: {list(self.available_models.keys())}")
    
    async def generate_response(
        self,
        prompt: str,
        task_type: str = "conversational",
        context: Optional[Dict[str, Any]] = None,
        user_id: str = "anonymous",
        prefer_local: bool = True,
        max_models: int = 3,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        responses = []
        errors = []

        # Select best models for this task
        selected_models = self._select_models_for_task(task_type, max_models)
        logger.info(f"Generating response with models: {[m['model'] for m in selected_models]}")

        for model_config in selected_models:
            if len(responses) >= max_models:
                break

            try:
                if model_config['provider'] == 'local':
                    response = await self._generate_local_response(
                        prompt, model_config, context, temperature
                    )
                else:
                    response = await self._generate_remote_response(
                        prompt, model_config, context, temperature
                    )

                if response:
                    responses.append(response)
                    # Update performance history per response
                    self._update_performance_history(response.model, True, response.response_time)
                else:
                    err_msg = f"{model_config['model']} returned no response"
                    errors.append(err_msg)
                    logger.warning(err_msg)

            except Exception as e:
                error_msg = f"{model_config['model']} failed: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                # Update failure count
                self.failure_counts[model_config['model']] = self.failure_counts.get(model_config['model'], 0) + 1

        # If no model succeeded
        if not responses:
            return {
                "success": False,
                "error": "All models failed: " + "; ".join(errors),
                "models_tried": [m['model'] for m in selected_models]
            }

        require_consensus = False  # Set this based on your logic

        # Merge responses
        merged_response = await self.response_merger.merge_responses(
            responses, task_type, require_consensus
        )

        # Calculate performance scores
        scores = self._calculate_performance_scores(responses)

        processing_time = time.time() - start_time
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.debug(f"Merged response: {merged_response}")

        return {
            "success": True,
            "text": merged_response['text'],
            "confidence": merged_response['confidence'],
            "sources": merged_response.get('sources', []),
            "scores": scores,
            "processing_time": processing_time,
            "models_used": [r.model for r in responses],
            "models_failed": [m['model'] for m in selected_models if m['model'] not in [r.model for r in responses]],
            "metadata": {
                'task_type': task_type,
                'merge_strategy': merged_response['strategy'],
                'consensus_achieved': merged_response.get('consensus', False)
            }
        }

    async def generate_response_wrapper(self, prompt: str, **kwargs):
        detected_task = self.detect_task_type(prompt)
        task_str = detected_task.value
        return await self.generate_response(prompt, task_type=task_str, **kwargs)

    
    async def _generate_multiple_responses(
        self,
        prompt: str,
        models: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> List[LLMResponse]:
        """Generate responses from multiple LLMs concurrently"""
        
        tasks = []
        
        for model_config in models:
            if model_config['provider'] == 'local':
                task = self._generate_local_response(prompt, model_config, context , temperature=0.7)
            else:
                task = self._generate_remote_response(prompt, model_config, context , temperature=0.7)
            
            tasks.append(task)
        
        # Execute all tasks concurrently with timeout
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_responses = []
            for result in results:
                if isinstance(result, LLMResponse):
                    valid_responses.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"LLM generation failed: {str(result)}")
            
            return valid_responses
            
        except Exception as e:
            logger.error(f"Multiple response generation error: {str(e)}")
            return []
    
    async def generate_response_stream(
        self,
        prompt: str,
        task_type: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with fallback to other models"""
        selected_models = self._select_models_for_task(task_type, 3)
        
        for model_config in selected_models:
            try:
                if model_config['provider'] == 'local':
                    async for chunk in self.local_llm.generate_stream(
                        prompt, model_config['model'], context, temperature
                    ):
                        yield chunk
                    return  # Success, stop trying
                else:
                    async for chunk in self.remote_llm.generate_stream(
                        prompt, model_config['model'], context, temperature
                    ):
                        yield chunk
                    return  # Success, stop trying
            except Exception as e:
                logger.warning(f"Streaming failed with {model_config['model']}: {str(e)}")
                # Try next model
        
        # If all models failed
        yield json.dumps({"error": "All streaming models failed"})
    
    def _select_models_for_task(self, task_type: str, max_models: int) -> List[Dict[str, Any]]:
        # Get preferred models list for the task_type string key, fallback to 'conversational'
        preferred_models = self.task_preferences.get(task_type, self.task_preferences.get('conversational', []))
        
        # Get free models based on weight >= 0.6 from llm_priority
        free_models_set = {entry['model'] for entry in self.llm_priority if entry['weight'] >= 0.6}
        
        # Filter preferred models by available free models, keep order
        filtered_models = [m for m in preferred_models if m in free_models_set]
        
        # Prepare model configs from llm_priority for those filtered models
        selected = []
        for model in filtered_models:
            for entry in self.llm_priority:
                if entry['model'] == model:
                    selected.append(entry)
                    break
            if len(selected) >= max_models:
                break

        # If no models found (empty), fallback to top free models by weight
        if not selected:
            sorted_free = sorted(
                [entry for entry in self.llm_priority if entry['weight'] >= 0.6],
                key=lambda e: e['weight'], reverse=True
            )
            selected = sorted_free[:max_models]

        return selected
    
    
    async def _generate_local_response(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        temperature: float
    ) -> Optional[LLMResponse]:
        """Generate response from local LLM with fallback handling"""
        try:
            start_time = time.time()
            result = await self.local_llm.generate(
                prompt, model_config['model'], context
            )
            
            if not result or "error" in result:
                return None
                
            response_time = time.time() - start_time
            quality_score = self._assess_quality(result['text'])
            
            return LLMResponse(
                text=result['text'],
                model=model_config['model'],
                provider='local',
                response_time=response_time,
                quality_score=quality_score,
                tokens_used=result.get('tokens_used', 0),
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Local LLM error ({model_config['model']}): {str(e)}")
            return None
    
    async def _generate_remote_response(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        temperature: float
    ) -> Optional[LLMResponse]:
        """Generate response from remote LLM with fallback handling"""
        try:
            start_time = time.time()
            result = await self.remote_llm.generate(
                prompt, model_config['model'], context
            )
            
            if not result or "error" in result:
                return None
                
            response_time = time.time() - start_time
            quality_score = self._assess_quality(result['text'])
            
            return LLMResponse(
                text=result['text'],
                model=model_config['model'],
                provider='remote',
                response_time=response_time,
                quality_score=quality_score,
                tokens_used=result.get('tokens_used', 0),
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Remote LLM error ({model_config['model']}): {str(e)}")
            return None
    
    def _assess_quality(self, text: str) -> float:
        """Assess response quality using heuristics (0.0-1.0)"""
        if not text:
            return 0.0
            
        # Length-based scoring
        word_count = len(text.split())
        length_score = min(word_count / 200, 1.0)  # Normalize to 200 words
        
        # Structure-based scoring
        sentence_count = len([s for s in text.split('.') if s.strip()])
        structure_score = min(sentence_count / 8, 1.0)  # Normalize to 8 sentences
        
        # Formatting-based scoring
        format_score = 0.0
        if '```' in text:  # Code blocks
            format_score += 0.3
        if any(line.strip().startswith(('- ', '* ', '1. ')) for line in text.split('\n')):
            format_score += 0.2  # Lists
        if '|' in text and '\n' in text:  # Potential table
            format_score += 0.1
        
        # Combined score with weights
        return min((
            length_score * 0.4 + 
            structure_score * 0.4 + 
            format_score * 0.2
        ), 1.0)
    
    def _calculate_performance_scores(
        self,
        responses: List[LLMResponse]
    ) -> Dict[str, float]:
        """Calculate performance scores for each model"""
        scores = {}
        
        for response in responses:
            # Combine quality and speed scores
            speed_score = max(0, 1 - (response.response_time / 15))  # Normalize to 15 seconds
            overall_score = (response.quality_score * 0.7) + (speed_score * 0.3)
            scores[response.model] = round(overall_score, 2)
        
        return scores
    
    def _update_performance_history(
        self,
        model_name: str,
        success: bool,
        response_time: float
    ):
        """Update performance history for model selection optimization"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {
                'scores': [],
                'response_times': [],
                'success_count': 0,
                'failure_count': 0,
                'total_count': 0
            }
        
        history = self.performance_history[model_name]
        history['total_count'] += 1
        
        if success:
            history['success_count'] += 1
            history['response_times'].append(response_time)
            # Keep only last 100 response times
            if len(history['response_times']) > 100:
                history['response_times'] = history['response_times'][-100:]
        else:
            history['failure_count'] += 1
    
    async def _check_available_models(self):
        """Check which models are available"""
        # Check local models
        local_models = await self.local_llm.get_available_models()
        
        # Check remote models
        remote_models = await self.remote_llm.get_available_models()
        
        # Combine all available models
        self.available_models = {**local_models, **remote_models}
        
        # Initialize failure counts
        for model in self.available_models:
            self.failure_counts.setdefault(model, 0)
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get available models and their status"""
        return {
            'models': list(self.available_models.keys()),
            'performance_history': self.performance_history,
            'failure_counts': self.failure_counts
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for LLM orchestrator"""
        return {
            'status': 'healthy' if self.available_models else 'degraded',
            'available_models': len(self.available_models),
            'local_llm_status': await self.local_llm.health_check(),
            'remote_llm_status': await self.remote_llm.health_check()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.local_llm.cleanup()
        await self.remote_llm.cleanup()