"""
Multi-LLM orchestration service with intelligent response merging
"""
from cachetools import TTLCache
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
import aiohttp
from dataclasses import dataclass

from core.logger import get_logger
from services.local_llm_service import LocalLLMService
from services.remote_llm_service import RemoteLLMService
from services.response_merger import ResponseMerger

logger = get_logger(__name__)

@dataclass
class LLMResponse:
    """LLM response with metadata"""
    text: str
    model: str
    provider: str
    response_time: float
    quality_score: float
    metadata: Dict[str, Any]

class LLMOrchestrator:
    """Orchestrates multiple LLMs for optimal response generation"""

    def __init__(self, settings):
        self.settings = settings
        self.local_llm = LocalLLMService(settings)
        self.remote_llm = RemoteLLMService(settings)
        self.response_merger = ResponseMerger(settings)

        # LLM priority and configuration
        self.llm_priority = [
            # Local LLMs (via Ollama)
            {"provider": "local", "model": "mistral", "weight": 1.0},

            # Remote LLMs (fallback)
            {"provider": "remote", "model": "together", "weight": 0.7},
            {"provider": "remote", "model": "openrouter", "weight": 0.6},
            {"provider": "remote", "model": "groq", "weight": 0.5},
            {"provider": "remote", "model": "huggingface", "weight": 0.4},
            {"provider": "remote", "model": "openai", "weight": 0.3},
            {"provider": "remote", "model": "anthropic", "weight": 0.2}
        ]

        # Task-specific model preferences
        self.task_preferences = {
            'code_generation': ['mistral', 'together', 'openai'],
            'code_fix': ['mistral', 'groq', 'together'],
            'code_review': ['openai', 'anthropic', 'together'],
            'research': ['mistral', 'groq', 'together'],
            'creative': ['openai', 'anthropic', 'together'],
            'analytical': ['openai', 'groq', 'together'],
            'conversational': ['mistral', 'groq', 'together']
        }

        # Performance tracking
        self.performance_history = {}
        self.available_models = {}

    
    async def initialize(self):
        """Initialize LLM services"""
        logger.info("Initializing LLM orchestrator...")
        
        # Initialize local LLM service
        await self.local_llm.initialize()
        
        # Initialize remote LLM service
        await self.remote_llm.initialize()
        
        # Check available models
        await self._check_available_models()
        
        logger.info("LLM orchestrator initialized successfully")
    
    async def generate_response(
        self,
        prompt: str,
        task_type: str = "conversational",
        context: Optional[Dict[str, Any]] = None,
        max_models: int = 3,
        require_consensus: bool = False,
        temperature=0.7,
        provider='ollama',
        model: Optional[str] = None  # <- ADD THIS
    ) -> Dict[str, Any]:
        """Generate response using multiple LLMs with intelligent merging"""
        
        start_time = time.time()
        
        try:
            # Select best models for this task
            selected_models = self._select_models_for_task(task_type, max_models)
            
            logger.info(f"Generating response with models: {[m['model'] for m in selected_models]}")
            
            # Generate responses from multiple LLMs
            responses = await self._generate_multiple_responses(
                prompt, selected_models, context
            )
            
            if not responses:
                raise Exception("No LLM responses generated")
            print(responses)
            # Merge responses intelligently
            merged_response = await self.response_merger.merge_responses(
                responses, task_type, require_consensus
            )
            
            # Calculate performance scores
            scores = self._calculate_performance_scores(responses)
            
            # Update performance history
            await self._update_performance_history(responses, scores)
            
            processing_time = time.time() - start_time
            
            return {
                'text': merged_response['text'],
                'confidence': merged_response['confidence'],
                'sources': merged_response['sources'],
                'scores': scores,
                'processing_time': processing_time,
                'models_used': [r.model for r in responses],
                'metadata': {
                    'task_type': task_type,
                    'merge_strategy': merged_response['strategy'],
                    'consensus_achieved': merged_response.get('consensus', False)
                }
            }
            
        except Exception as e:
            logger.error(f"LLM orchestration error: {str(e)}")
            raise
    
    async def generate_response_stream(
        self,
        prompt: str,
        task_type: str = "conversational",
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from best available LLM"""
        
        try:
            # Select best single model for streaming
            selected_models = self._select_models_for_task(task_type, 1)
            
            if not selected_models:
                yield "Error: No available models for streaming"
                return
            
            model_config = selected_models[0]
            
            # Stream from selected model
            if model_config['provider'] == 'local':
                async for chunk in self.local_llm.generate_stream(
                    prompt, model_config['model'], context
                ):
                    yield chunk
            else:
                async for chunk in self.remote_llm.generate_stream(
                    prompt, model_config['model'], context
                ):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"Error: {str(e)}"
    
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
                task = self._generate_local_response(prompt, model_config, context)
            else:
                task = self._generate_remote_response(prompt, model_config, context)
            
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
    
    async def _generate_local_response(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Optional[LLMResponse]:
        """Generate response from local LLM"""
        
        try:
            start_time = time.time()
            
            result = await self.local_llm.generate(
                prompt, model_config['model'], context
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                text=result['text'],
                model=model_config['model'],
                provider='local',
                response_time=response_time,
                quality_score=self._assess_quality(result['text']),
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Local LLM error ({model_config['model']}): {str(e)}")
            return None
    
    async def _generate_remote_response(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Optional[LLMResponse]:
        """Generate response from remote LLM"""
        
        try:
            start_time = time.time()
            
            result = await self.remote_llm.generate(
                prompt, model_config['model'], context
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                text=result['text'],
                model=model_config['model'],
                provider='remote',
                response_time=response_time,
                quality_score=self._assess_quality(result['text']),
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Remote LLM error ({model_config['model']}): {str(e)}")
            return None
    
    def _select_models_for_task(
        self,
        task_type: str,
        max_models: int
    ) -> List[Dict[str, Any]]:
        """Select best models for specific task type"""
        
        # Get task-specific preferences
        preferred_models = self.task_preferences.get(task_type, [])
        
        # Filter available models by preferences
        selected = []
        
        # First, add preferred models if available
        for model_name in preferred_models:
            for model_config in self.llm_priority:
                if (model_config['model'] == model_name and 
                    model_config['model'] in self.available_models):
                    selected.append(model_config)
                    break
        
        # Fill remaining slots with best available models
        for model_config in self.llm_priority:
            if (len(selected) >= max_models):
                break
            
            if (model_config not in selected and 
                model_config['model'] in self.available_models):
                selected.append(model_config)
        
        return selected[:max_models]
    
    def _assess_quality(self, text: str) -> float:
        """Assess response quality (simple heuristic)"""
        if not text:
            return 0.0
        
        # Basic quality metrics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Quality factors
        length_score = min(word_count / 100, 1.0)  # Normalize to 100 words
        structure_score = min(sentence_count / 5, 1.0)  # Normalize to 5 sentences
        
        # Check for code blocks, lists, etc.
        format_score = 0.0
        if '```' in text:
            format_score += 0.2
        if any(line.strip().startswith(('- ', '* ', '1. ')) for line in text.split('\n')):
            format_score += 0.1
        
        return min((length_score + structure_score + format_score) / 2, 1.0)
    
    def _calculate_performance_scores(
        self,
        responses: List[LLMResponse]
    ) -> Dict[str, float]:
        """Calculate performance scores for each model"""
        
        scores = {}
        
        for response in responses:
            # Combine quality and speed scores
            speed_score = max(0, 1 - (response.response_time / 10))  # Normalize to 10 seconds
            quality_score = response.quality_score
            
            # Weighted combination
            overall_score = (quality_score * 0.7) + (speed_score * 0.3)
            
            scores[response.model] = overall_score
        
        return scores
    
    async def _update_performance_history(
        self,
        responses: List[LLMResponse],
        scores: Dict[str, float]
    ):
        """Update performance history for model selection optimization"""
        
        for response in responses:
            model_name = response.model
            
            if model_name not in self.performance_history:
                self.performance_history[model_name] = {
                    'scores': [],
                    'response_times': [],
                    'success_count': 0,
                    'total_count': 0
                }
            
            history = self.performance_history[model_name]
            history['scores'].append(scores[model_name])
            history['response_times'].append(response.response_time)
            history['success_count'] += 1
            history['total_count'] += 1
            
            # Keep only last 100 entries
            if len(history['scores']) > 100:
                history['scores'] = history['scores'][-100:]
                history['response_times'] = history['response_times'][-100:]
    
    async def _check_available_models(self):
        """Check which models are available"""
        
        # Check local models
        local_models = await self.local_llm.get_available_models()
        
        # Check remote models
        remote_models = await self.remote_llm.get_available_models()
        
        # Combine all available models
        self.available_models = {**local_models, **remote_models}
        
        logger.info(f"Available models: {list(self.available_models.keys())}")
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get available models and their status"""
        return {
            'local': await self.local_llm.get_available_models(),
            'remote': await self.remote_llm.get_available_models(),
            'performance_history': self.performance_history
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for LLM orchestrator"""
        return {
            'status': 'healthy',
            'available_models': len(self.available_models),
            'local_llm_status': await self.local_llm.health_check(),
            'remote_llm_status': await self.remote_llm.health_check()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.local_llm.cleanup()
        await self.remote_llm.cleanup()
