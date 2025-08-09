"""
Local LLM service for Ollama, LlamaCPP, and GPT4All
"""

import asyncio
import aiohttp
import json
import subprocess
from typing import Dict, List, Optional, Any, AsyncGenerator

from core.logger import get_logger

logger = get_logger(__name__)

class LocalLLMService:
    """Service for managing local LLM providers"""
    
    def __init__(self, settings):
        self.settings = settings
        self.session = None
        
        # Local LLM configurations
        self.providers = {
            'ollama': {
                'url': settings.OLLAMA_BASE_URL,
                'endpoint': '/api/generate',
                'health_endpoint': '/api/tags',
                'available': False
            },
            'llamacpp': {
                'url': settings.LLAMACPP_HOST,
                'endpoint': '/completion',
                'health_endpoint': '/health',
                'available': False
            },
            'gpt4all': {
                'url': settings.GPT4ALL_HOST,
                'endpoint': '/v1/chat/completions',
                'health_endpoint': '/health',
                'available': False
            }
        }
        
        # Model configurations
        self.model_configs = {
            'ollama': {
                'default_model': 'mistral',
                'code_model': 'codellama:7b',
                'chat_model': 'mistral'
            },
            'llamacpp': {
                'default_model': 'llama-2-7b-chat',
                'max_tokens': 2048,
                'temperature': 0.7
            },
            'gpt4all': {
                'default_model': 'gpt4all-j',
                'max_tokens': 2048,
                'temperature': 0.7
            }
        }
    
    async def initialize(self):
        """Initialize local LLM service"""
        logger.info("Initializing local LLM service...")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        )
        
        # Check availability of each provider
        await self._check_providers()
        
        logger.info("Local LLM service initialized")
    
    async def generate(
        self,
        prompt: str,
        model: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from local LLM"""
        
        if model not in self.providers:
            raise ValueError(f"Unknown local model: {model}")
        
        provider = self.providers[model]
        if not provider['available']:
            raise Exception(f"Local model {model} is not available")
        
        try:
            if model == 'ollama':
                return await self._generate_ollama(prompt, context)
            elif model == 'llamacpp':
                return await self._generate_llamacpp(prompt, context)
            elif model == 'gpt4all':
                return await self._generate_gpt4all(prompt, context)
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            logger.error(f"Local LLM generation error ({model}): {str(e)}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        model: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from local LLM"""
        
        if model not in self.providers:
            raise ValueError(f"Unknown local model: {model}")
        
        provider = self.providers[model]
        if not provider['available']:
            raise Exception(f"Local model {model} is not available")
        
        try:
            if model == 'ollama':
                async for chunk in self._generate_ollama_stream(prompt, context):
                    yield chunk
            elif model == 'llamacpp':
                async for chunk in self._generate_llamacpp_stream(prompt, context):
                    yield chunk
            elif model == 'gpt4all':
                async for chunk in self._generate_gpt4all_stream(prompt, context):
                    yield chunk
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            logger.error(f"Local LLM streaming error ({model}): {str(e)}")
            yield f"Error: {str(e)}"
    
    async def _generate_ollama(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from Ollama"""
        
        config = self.model_configs['ollama']
        
        # Select appropriate model based on context
        model_name = config['default_model']
        if context and context.get('task_type') == 'code_generation':
            model_name = config['code_model']
        
        payload = {
            'model': model_name,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40
            }
        }
        
        url = f"{self.providers['ollama']['url']}{self.providers['ollama']['endpoint']}"
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
            if response.status == 200:
                result = await response.json()
                return {
                    'text': result.get('response', ''),
                    'metadata': {
                        'model': model_name,
                        'provider': 'ollama',
                        'tokens': result.get('eval_count', 0),
                        'prompt_tokens': result.get('prompt_eval_count', 0)
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"Ollama API error: {response.status} - {error_text}")
    
    async def _generate_ollama_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama"""
        
        config = self.model_configs['ollama']
        model_name = config['default_model']
        
        payload = {
            'model': model_name,
            'prompt': prompt,
            'stream': True,
            'options': {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40
            }
        }
        
        url = f"{self.providers['ollama']['url']}{self.providers['ollama']['endpoint']}"
        
        async with self.session.post(url, json=payload) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def _generate_llamacpp(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from LlamaCPP"""
        
        config = self.model_configs['llamacpp']
        
        payload = {
            'prompt': prompt,
            'n_predict': config['max_tokens'],
            'temperature': config['temperature'],
            'top_p': 0.9,
            'top_k': 40,
            'stop': ['</s>', 'Human:', 'Assistant:']
        }
        
        url = f"{self.providers['llamacpp']['url']}{self.providers['llamacpp']['endpoint']}"
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    'text': result.get('content', ''),
                    'metadata': {
                        'model': config['default_model'],
                        'provider': 'llamacpp',
                        'tokens_predicted': result.get('tokens_predicted', 0),
                        'tokens_evaluated': result.get('tokens_evaluated', 0)
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"LlamaCPP API error: {response.status} - {error_text}")
    
    async def _generate_llamacpp_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from LlamaCPP"""
        
        config = self.model_configs['llamacpp']
        
        payload = {
            'prompt': prompt,
            'n_predict': config['max_tokens'],
            'temperature': config['temperature'],
            'stream': True,
            'stop': ['</s>', 'Human:', 'Assistant:']
        }
        
        url = f"{self.providers['llamacpp']['url']}{self.providers['llamacpp']['endpoint']}"
        
        async with self.session.post(url, json=payload) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'content' in data:
                            yield data['content']
                        if data.get('stop', False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def _generate_gpt4all(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from GPT4All"""
        
        config = self.model_configs['gpt4all']
        
        payload = {
            'model': config['default_model'],
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': config['max_tokens'],
            'temperature': config['temperature'],
            'stream': False
        }
        
        url = f"{self.providers['gpt4all']['url']}{self.providers['gpt4all']['endpoint']}"
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                return {
                    'text': content,
                    'metadata': {
                        'model': config['default_model'],
                        'provider': 'gpt4all',
                        'usage': result.get('usage', {})
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"GPT4All API error: {response.status} - {error_text}")
    
    async def _generate_gpt4all_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from GPT4All"""
        
        config = self.model_configs['gpt4all']
        
        payload = {
            'model': config['default_model'],
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': config['max_tokens'],
            'temperature': config['temperature'],
            'stream': True
        }
        
        url = f"{self.providers['gpt4all']['url']}{self.providers['gpt4all']['endpoint']}"
        
        async with self.session.post(url, json=payload) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
    
    async def _check_providers(self) -> None:
        for provider_name, config in self.providers.items():
            health_url = f"{config['url']}{config['health_endpoint']}"
            print(health_url)
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with self.session.get(health_url, timeout=timeout) as response:
                    if response.status == 200:
                        config['available'] = True
                        logger.info(f"✅ {provider_name} is available")
                    else:
                        config['available'] = False
                        logger.warning(f"❌ {provider_name} health check failed: HTTP {response.status}")
            except Exception as e:
                config['available'] = False
                logger.warning(f"❌ {provider_name} is not available: {e}")
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get available local models"""
        available = {}
        
        for provider_name, config in self.providers.items():
            if config['available']:
                available[provider_name] = {
                    'status': 'available',
                    'url': config['url'],
                    'models': await self._get_provider_models(provider_name)
                }
            else:
                available[provider_name] = {
                    'status': 'unavailable',
                    'url': config['url'],
                    'models': []
                }
        
        return available
    
    async def _get_provider_models(self, provider_name: str) -> List[str]:
        """Get available models for specific provider"""
        
        if provider_name == 'ollama':
            try:
                url = f"{self.providers['ollama']['url']}/api/tags"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['name'] for model in data.get('models', [])]
            except Exception as e:
                logger.error(f"Error getting Ollama models: {str(e)}")
        
        # For other providers, return configured models
        return list(self.model_configs.get(provider_name, {}).values())
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for local LLM service"""
        await self._check_providers()
        
        return {
            'status': 'healthy' if any(p['available'] for p in self.providers.values()) else 'degraded',
            'providers': {
                name: {'available': config['available'], 'url': config['url']}
                for name, config in self.providers.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
