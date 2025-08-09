"""
Remote LLM service for Together.ai, OpenRouter, Groq, HuggingFace, OpenAI, Anthropic
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator

from core.logger import get_logger

logger = get_logger(__name__)

class RemoteLLMService:
    """Service for managing remote LLM providers"""
    
    def __init__(self, settings):
        self.settings = settings
        self.session = None
        
        # Remote LLM configurations
        self.providers = {
            'together': {
                'url': 'https://api.together.xyz/v1/chat/completions',
                'api_key': settings.TOGETHER_API_KEY,
                'models': ['meta-llama/Llama-3-8b-chat-hf', 'meta-llama/Llama-3-70b-chat-hf'],
                'available': bool(settings.TOGETHER_API_KEY)
            },
            'openrouter': {
                'url': 'https://openrouter.ai/api/v1/chat/completions',
                'api_key': settings.OPENROUTER_API_KEY,
                'models': ['openai/gpt-3.5-turbo', 'anthropic/claude-3-sonnet'],
                'available': bool(settings.OPENROUTER_API_KEY)
            },
            'groq': {
                'url': 'https://api.groq.com/openai/v1/chat/completions',
                'api_key': settings.GROQ_API_KEY,
                'models': ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768'],
                'available': bool(settings.GROQ_API_KEY)
            },
            'huggingface': {
                'url': 'https://api-inference.huggingface.co/models',
                'api_key': settings.HUGGINGFACE_API_KEY,
                'models': ['microsoft/DialoGPT-large', 'facebook/blenderbot-400M-distill'],
                'available': bool(settings.HUGGINGFACE_API_KEY)
            },
            'openai': {
                'url': 'https://api.openai.com/v1/chat/completions',
                'api_key': settings.OPENAI_API_KEY,
                'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
                'available': bool(settings.OPENAI_API_KEY)
            },
            'anthropic': {
                'url': 'https://api.anthropic.com/v1/messages',
                'api_key': settings.ANTHROPIC_API_KEY,
                'models': ['claude-3-sonnet-20240229', 'claude-3-opus-20240229'],
                'available': bool(settings.ANTHROPIC_API_KEY)
            }
        }
        
        # Task-specific model selection
        self.task_models = {
            'code_generation': {
                'together': 'meta-llama/Llama-3-70b-chat-hf',
                'openrouter': 'openai/gpt-4',
                'groq': 'llama3-70b-8192',
                'openai': 'gpt-4',
                'anthropic': 'claude-3-opus-20240229'
            },
            'creative': {
                'together': 'meta-llama/Llama-3-8b-chat-hf',
                'openrouter': 'anthropic/claude-3-sonnet',
                'openai': 'gpt-4',
                'anthropic': 'claude-3-sonnet-20240229'
            },
            'analytical': {
                'together': 'meta-llama/Llama-3-70b-chat-hf',
                'groq': 'mixtral-8x7b-32768',
                'openai': 'gpt-4-turbo',
                'anthropic': 'claude-3-opus-20240229'
            }
        }
    
    async def initialize(self):
        """Initialize remote LLM service"""
        logger.info("Initializing remote LLM service...")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        
        # Check availability
        await self._check_providers()
        
        logger.info("Remote LLM service initialized")
    
    async def generate(
        self,
        prompt: str,
        model: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from remote LLM"""
        
        if model not in self.providers:
            raise ValueError(f"Unknown remote model: {model}")
        
        provider = self.providers[model]
        if not provider['available']:
            raise Exception(f"Remote model {model} is not available")
        
        try:
            if model == 'together':
                return await self._generate_together(prompt, context)
            elif model == 'openrouter':
                return await self._generate_openrouter(prompt, context)
            elif model == 'groq':
                return await self._generate_groq(prompt, context)
            elif model == 'huggingface':
                return await self._generate_huggingface(prompt, context)
            elif model == 'openai':
                return await self._generate_openai(prompt, context)
            elif model == 'anthropic':
                return await self._generate_anthropic(prompt, context)
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            logger.error(f"Remote LLM generation error ({model}): {str(e)}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        model: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from remote LLM"""
        
        if model not in self.providers:
            raise ValueError(f"Unknown remote model: {model}")
        
        provider = self.providers[model]
        if not provider['available']:
            raise Exception(f"Remote model {model} is not available")
        
        try:
            if model == 'together':
                async for chunk in self._generate_together_stream(prompt, context):
                    yield chunk
            elif model == 'openrouter':
                async for chunk in self._generate_openrouter_stream(prompt, context):
                    yield chunk
            elif model == 'groq':
                async for chunk in self._generate_groq_stream(prompt, context):
                    yield chunk
            elif model == 'openai':
                async for chunk in self._generate_openai_stream(prompt, context):
                    yield chunk
            elif model == 'anthropic':
                async for chunk in self._generate_anthropic_stream(prompt, context):
                    yield chunk
            else:
                raise ValueError(f"Streaming not supported for model: {model}")
                
        except Exception as e:
            logger.error(f"Remote LLM streaming error ({model}): {str(e)}")
            yield f"Error: {str(e)}"
    
    async def _generate_together(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from Together.ai"""
        
        # Select model based on task
        model_name = 'meta-llama/Llama-3-8b-chat-hf'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('together', model_name)
        
        headers = {
            'Authorization': f"Bearer {self.providers['together']['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2048,
            'stream': False
        }
        
        async with self.session.post(
            self.providers['together']['url'],
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                return {
                    'text': content,
                    'metadata': {
                        'model': model_name,
                        'provider': 'together',
                        'usage': result.get('usage', {})
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"Together.ai API error: {response.status} - {error_text}")
    
    async def _generate_together_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Together.ai"""
        
        model_name = 'meta-llama/Llama-3-8b-chat-hf'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('together', model_name)
        
        headers = {
            'Authorization': f"Bearer {self.providers['together']['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2048,
            'stream': True
        }
        
        async with self.session.post(
            self.providers['together']['url'],
            headers=headers,
            json=payload
        ) as response:
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
    
    async def _generate_openrouter(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from OpenRouter"""
        
        model_name = 'openai/gpt-3.5-turbo'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('openrouter', model_name)
        
        headers = {
            'Authorization': f"Bearer {self.providers['openrouter']['api_key']}",
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://damn-bot.ai',
            'X-Title': 'DAMN BOT'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2048
        }
        
        async with self.session.post(
            self.providers['openrouter']['url'],
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                return {
                    'text': content,
                    'metadata': {
                        'model': model_name,
                        'provider': 'openrouter',
                        'usage': result.get('usage', {})
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
    
    async def _generate_openrouter_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenRouter"""
        
        model_name = 'openai/gpt-3.5-turbo'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('openrouter', model_name)
        
        headers = {
            'Authorization': f"Bearer {self.providers['openrouter']['api_key']}",
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://damn-bot.ai',
            'X-Title': 'DAMN BOT'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2048,
            'stream': True
        }
        
        async with self.session.post(
            self.providers['openrouter']['url'],
            headers=headers,
            json=payload
        ) as response:
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
    
    async def _generate_groq(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from Groq"""
        
        model_name = 'llama3-8b-8192'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('groq', model_name)
        
        headers = {
            'Authorization': f"Bearer {self.providers['groq']['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2048
        }
        
        async with self.session.post(
            self.providers['groq']['url'],
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                return {
                    'text': content,
                    'metadata': {
                        'model': model_name,
                        'provider': 'groq',
                        'usage': result.get('usage', {})
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"Groq API error: {response.status} - {error_text}")
    
    async def _generate_groq_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Groq"""
        
        model_name = 'llama3-8b-8192'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('groq', model_name)
        
        headers = {
            'Authorization': f"Bearer {self.providers['groq']['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2048,
            'stream': True
        }
        
        async with self.session.post(
            self.providers['groq']['url'],
            headers=headers,
            json=payload
        ) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        if line == 'data: [DONE]':
                            break
                        try:
                            data = json.loads(line[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
    
    async def _generate_huggingface(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from HuggingFace"""
        
        model_name = 'microsoft/DialoGPT-large'
        
        headers = {
            'Authorization': f"Bearer {self.providers['huggingface']['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'inputs': prompt,
            'parameters': {
                'max_length': 500,
                'temperature': 0.7,
                'do_sample': True
            }
        }
        
        url = f"{self.providers['huggingface']['url']}/{model_name}"
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                # HuggingFace returns different formats
                if isinstance(result, list) and result:
                    text = result[0].get('generated_text', '')
                else:
                    text = str(result)
                
                return {
                    'text': text,
                    'metadata': {
                        'model': model_name,
                        'provider': 'huggingface'
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"HuggingFace API error: {response.status} - {error_text}")
    
    async def _generate_openai(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from OpenAI"""
        
        model_name = 'gpt-3.5-turbo'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('openai', model_name)
        
        headers = {
            'Authorization': f"Bearer {self.providers['openai']['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2048
        }
        
        async with self.session.post(
            self.providers['openai']['url'],
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                return {
                    'text': content,
                    'metadata': {
                        'model': model_name,
                        'provider': 'openai',
                        'usage': result.get('usage', {})
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"OpenAI API error: {response.status} - {error_text}")
    
    async def _generate_openai_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenAI"""
        
        model_name = 'gpt-3.5-turbo'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('openai', model_name)
        
        headers = {
            'Authorization': f"Bearer {self.providers['openai']['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 2048,
            'stream': True
        }
        
        async with self.session.post(
            self.providers['openai']['url'],
            headers=headers,
            json=payload
        ) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        if line == 'data: [DONE]':
                            break
                        try:
                            data = json.loads(line[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
    
    async def _generate_anthropic(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response from Anthropic"""
        
        model_name = 'claude-3-sonnet-20240229'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('anthropic', model_name)
        
        headers = {
            'x-api-key': self.providers['anthropic']['api_key'],
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 2048
        }
        
        async with self.session.post(
            self.providers['anthropic']['url'],
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result['content'][0]['text']
                return {
                    'text': content,
                    'metadata': {
                        'model': model_name,
                        'provider': 'anthropic',
                        'usage': result.get('usage', {})
                    }
                }
            else:
                error_text = await response.text()
                raise Exception(f"Anthropic API error: {response.status} - {error_text}")
    
    async def _generate_anthropic_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Anthropic"""
        
        model_name = 'claude-3-sonnet-20240229'
        if context and context.get('task_type') in self.task_models:
            model_name = self.task_models[context['task_type']].get('anthropic', model_name)
        
        headers = {
            'x-api-key': self.providers['anthropic']['api_key'],
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 2048,
            'stream': True
        }
        
        async with self.session.post(
            self.providers['anthropic']['url'],
            headers=headers,
            json=payload
        ) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if 'delta' in data and 'text' in data['delta']:
                                yield data['delta']['text']
                        except json.JSONDecodeError:
                            continue
    
    async def _check_providers(self):
        """Check availability of remote providers"""
        
        for provider_name, config in self.providers.items():
            if not config['api_key']:
                config['available'] = False
                logger.warning(f"❌ {provider_name} API key not provided")
                continue
            
            try:
                # Simple test request
                if provider_name == 'anthropic':
                    headers = {
                        'x-api-key': config['api_key'],
                        'Content-Type': 'application/json',
                        'anthropic-version': '2023-06-01'
                    }
                    test_payload = {
                        'model': 'claude-3-sonnet-20240229',
                        'messages': [{'role': 'user', 'content': 'test'}],
                        'max_tokens': 10
                    }
                else:
                    headers = {
                        'Authorization': f"Bearer {config['api_key']}",
                        'Content-Type': 'application/json'
                    }
                    test_payload = {
                        'model': config['models'][0],
                        'messages': [{'role': 'user', 'content': 'test'}],
                        'max_tokens': 10
                    }
                
                async with self.session.post(
                    config['url'],
                    headers=headers,
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in [200, 400, 401]:  # 400/401 means API is working
                        config['available'] = True
                        logger.info(f"✅ {provider_name} is available")
                    else:
                        config['available'] = False
                        logger.warning(f"❌ {provider_name} test failed: {response.status}")
                        
            except Exception as e:
                config['available'] = False
                logger.warning(f"❌ {provider_name} is not available: {str(e)}")
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get available remote models"""
        available = {}
        
        for provider_name, config in self.providers.items():
            available[provider_name] = {
                'status': 'available' if config['available'] else 'unavailable',
                'url': config['url'],
                'models': config['models'] if config['available'] else [],
                'has_api_key': bool(config['api_key'])
            }
        
        return available
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for remote LLM service"""
        await self._check_providers()
        
        return {
            'status': 'healthy' if any(p['available'] for p in self.providers.values()) else 'degraded',
            'providers': {
                name: {
                    'available': config['available'],
                    'has_api_key': bool(config['api_key'])
                }
                for name, config in self.providers.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
