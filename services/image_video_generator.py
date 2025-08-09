"""
Image and Video Generation Service - Free and local media generation
"""

import asyncio
import aiohttp
import base64
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import re

from core.config import get_settings
from core.logger import get_logger
from services.llm_orchestrator import LLMOrchestrator

logger = get_logger(__name__)

class ImageVideoGenerator:
    """Service for generating images and videos using free/local providers"""
    
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        self.settings = get_settings()
        self.llm_orchestrator = llm_orchestrator
        
        # Media generation providers
        self.image_providers = {
            "huggingface": {
                "enabled": bool(self.settings.HUGGINGFACE_API_KEY),
                "api_key": self.settings.HUGGINGFACE_API_KEY,
                "endpoint": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
                "priority": 1,
                "free_tier": True,
                "formats": ["png", "jpg"]
            },
            "replicate": {
                "enabled": bool(self.settings.REPLICATE_API_TOKEN),
                "api_key": self.settings.REPLICATE_API_TOKEN,
                "endpoint": "https://api.replicate.com/v1/predictions",
                "priority": 2,
                "free_tier": True,
                "formats": ["png", "jpg"]
            },
            "automatic1111": {
                "enabled": True,  # Local installation
                "endpoint": f"{self.settings.AUTOMATIC1111_URL}/sdapi/v1/txt2img",
                "priority": 3,
                "free_tier": True,
                "local": True,
                "formats": ["png", "jpg"]
            }
        }
        
        self.video_providers = {
            "replicate_video": {
                "enabled": bool(self.settings.REPLICATE_API_TOKEN),
                "api_key": self.settings.REPLICATE_API_TOKEN,
                "endpoint": "https://api.replicate.com/v1/predictions",
                "model": "anotherjesse/zeroscope-v2-xl",
                "priority": 1,
                "free_tier": True,
                "formats": ["mp4"]
            }
        }
        
        # Media detection patterns
        self.image_patterns = [
            r"generate.*image", r"create.*image", r"draw.*", r"picture.*of",
            r"photo.*of", r"illustration.*of", r"artwork.*of", r"design.*",
            r"logo.*", r"icon.*", r"banner.*", r"poster.*"
        ]
        
        self.video_patterns = [
            r"generate.*video", r"create.*video", r"animate.*", r"movie.*of",
            r"clip.*of", r"animation.*of", r"motion.*", r"cinematic.*"
        ]
        
        # Style presets
        self.image_styles = {
            "photorealistic": "photorealistic, high quality, detailed, 8k resolution",
            "artistic": "artistic, painting style, creative, expressive",
            "cartoon": "cartoon style, animated, colorful, fun",
            "sketch": "pencil sketch, hand drawn, artistic, black and white",
            "digital_art": "digital art, modern, stylized, vibrant colors",
            "vintage": "vintage style, retro, aged, classic",
            "minimalist": "minimalist, clean, simple, modern design",
            "fantasy": "fantasy art, magical, mystical, ethereal"
        }
        
        self.video_styles = {
            "cinematic": "cinematic, professional, high quality, dramatic lighting",
            "documentary": "documentary style, realistic, natural lighting",
            "animated": "animated style, smooth motion, colorful",
            "artistic": "artistic, creative, experimental, unique perspective"
        }
        
        # Generation cache
        self.generation_cache = {}
        
        # HTTP session
        self.session = None
    
    async def initialize(self):
        """Initialize the image/video generation service"""
        try:
            logger.info("Initializing Image/Video Generation Service...")
            
            # Create media directory
            Path(self.settings.GENERATED_MEDIA_DIR).mkdir(parents=True, exist_ok=True)
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes for media generation
            )
            
            # Test providers
            await self._test_providers()
            
            logger.info("✅ Image/Video Generation Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Image/Video Generation Service: {str(e)}")
            raise
    
    async def detect_and_generate_media(
        self,
        user_message: str,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """Detect media generation intent and generate media"""
        try:
            # Detect media type
            media_detection = await self._detect_media_type(user_message)
            
            if not media_detection.get("is_media_request"):
                return {"is_media_request": False}
            
            media_type = media_detection["media_type"]
            
            # Extract generation parameters
            params = await self._extract_generation_params(user_message, media_type)
            
            # Generate media
            if media_type == "image":
                result = await self.generate_image(
                    prompt=params.get("prompt", user_message),
                    style=params.get("style", "photorealistic"),
                    size=params.get("size", "1024x1024"),
                    user_id=user_id
                )
            elif media_type == "video":
                result = await self.generate_video(
                    prompt=params.get("prompt", user_message),
                    style=params.get("style", "cinematic"),
                    duration=params.get("duration", 5),
                    user_id=user_id
                )
            else:
                return {"is_media_request": False, "error": f"Unsupported media type: {media_type}"}
            
            result.update({
                "is_media_request": True,
                "media_type": media_type,
                "detection_confidence": media_detection.get("confidence", 0.0)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Media detection and generation failed: {str(e)}")
            return {
                "is_media_request": False,
                "error": str(e)
            }
    
    async def generate_image(
        self,
        prompt: str,
        style: str = "photorealistic",
        size: str = "1024x1024",
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """Generate image from text prompt"""
        try:
            start_time = time.time()
            logger.info(f"Generating image: {prompt[:100]}...")
            
            # Check cache
            cache_key = self._generate_cache_key(prompt, style, size, "image")
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Enhance prompt with style
            enhanced_prompt = await self._enhance_image_prompt(prompt, style)
            
            # Try providers in order of priority
            for provider_name, provider_config in sorted(
                self.image_providers.items(),
                key=lambda x: x[1]["priority"]
            ):
                if not provider_config["enabled"]:
                    continue
                
                try:
                    result = await self._generate_image_with_provider(
                        provider_name, provider_config, enhanced_prompt, size
                    )
                    
                    if result.get("success"):
                        # Save to file system
                        image_path = await self._save_generated_image(
                            result["image_data"], user_id, provider_name
                        )
                        
                        final_result = {
                            "success": True,
                            "image_path": image_path,
                            "provider_used": provider_name,
                            "enhanced_prompt": enhanced_prompt,
                            "generation_time": time.time() - start_time,
                            "style": style,
                            "size": size
                        }
                        
                        # Cache result
                        self._cache_result(cache_key, final_result)
                        
                        logger.info(f"Image generated successfully using {provider_name}")
                        return final_result
                    
                except Exception as e:
                    logger.warning(f"Image generation failed with {provider_name}: {str(e)}")
                    continue
            
            return {
                "success": False,
                "error": "All image generation providers failed"
            }
            
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_video(
        self,
        prompt: str,
        style: str = "cinematic",
        duration: int = 5,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """Generate video from text prompt"""
        try:
            start_time = time.time()
            logger.info(f"Generating video: {prompt[:100]}...")
            
            # Check cache
            cache_key = self._generate_cache_key(prompt, style, str(duration), "video")
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Enhance prompt with style
            enhanced_prompt = await self._enhance_video_prompt(prompt, style)
            
            # Try providers in order of priority
            for provider_name, provider_config in sorted(
                self.video_providers.items(),
                key=lambda x: x[1]["priority"]
            ):
                if not provider_config["enabled"]:
                    continue
                
                try:
                    result = await self._generate_video_with_provider(
                        provider_name, provider_config, enhanced_prompt, duration
                    )
                    
                    if result.get("success"):
                        # Save to file system
                        video_path = await self._save_generated_video(
                            result["video_data"], user_id, provider_name
                        )
                        
                        final_result = {
                            "success": True,
                            "video_path": video_path,
                            "provider_used": provider_name,
                            "enhanced_prompt": enhanced_prompt,
                            "generation_time": time.time() - start_time,
                            "style": style,
                            "duration": duration
                        }
                        
                        # Cache result
                        self._cache_result(cache_key, final_result)
                        
                        logger.info(f"Video generated successfully using {provider_name}")
                        return final_result
                    
                except Exception as e:
                    logger.warning(f"Video generation failed with {provider_name}: {str(e)}")
                    continue
            
            return {
                "success": False,
                "error": "All video generation providers failed"
            }
            
        except Exception as e:
            logger.error(f"Video generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _detect_media_type(self, message: str) -> Dict[str, Any]:
        """Detect media generation type from message"""
        try:
            message_lower = message.lower()
            
            # Pattern matching
            image_score = sum(1 for pattern in self.image_patterns if re.search(pattern, message_lower))
            video_score = sum(1 for pattern in self.video_patterns if re.search(pattern, message_lower))
            
            if image_score == 0 and video_score == 0:
                # Use LLM for subtle detection
                llm_detection = await self._llm_media_detection(message)
                if llm_detection.get("is_media") and llm_detection.get("confidence", 0) > 0.6:
                    return {
                        "is_media_request": True,
                        "media_type": llm_detection.get("media_type", "image"),
                        "confidence": llm_detection.get("confidence", 0.7)
                    }
                else:
                    return {"is_media_request": False}
            
            if image_score > video_score:
                return {
                    "is_media_request": True,
                    "media_type": "image",
                    "confidence": min(image_score / len(self.image_patterns), 1.0)
                }
            else:
                return {
                    "is_media_request": True,
                    "media_type": "video",
                    "confidence": min(video_score / len(self.video_patterns), 1.0)
                }
                
        except Exception as e:
            logger.error(f"Media type detection failed: {str(e)}")
            return {"is_media_request": False, "error": str(e)}
    
    async def _llm_media_detection(self, message: str) -> Dict[str, Any]:
        """Use LLM for media detection"""
        try:
            detection_prompt = f"""
            Analyze this message to determine if the user wants to generate visual media:
            
            Message: "{message}"
            
            Media types:
            - image: Photos, pictures, illustrations, artwork, logos, designs
            - video: Videos, animations, clips, movies, motion graphics
            
            Respond in JSON format:
            {{
                "is_media": boolean,
                "media_type": "image" or "video",
                "confidence": float (0.0-1.0),
                "reasoning": "explanation"
            }}
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=detection_prompt,
                task_type="analytical",
                context={"media_detection": True}
            )
            
            try:
                return json.loads(response.get("text", "{}"))
            except json.JSONDecodeError:
                return {"is_media": False, "confidence": 0.0}
                
        except Exception as e:
            logger.error(f"LLM media detection failed: {str(e)}")
            return {"is_media": False, "confidence": 0.0}
    
    async def _extract_generation_params(
        self,
        message: str,
        media_type: str
    ) -> Dict[str, Any]:
        """Extract generation parameters from message"""
        try:
            extraction_prompt = f"""
            Extract {media_type} generation parameters from this message:
            
            Message: "{message}"
            
            Extract:
            1. Main subject/prompt (clean description)
            2. Style preference (if mentioned)
            3. Size/dimensions (if mentioned)
            4. Duration (for video, if mentioned)
            5. Any specific requirements
            
            Available styles:
            - Image: {list(self.image_styles.keys())}
            - Video: {list(self.video_styles.keys())}
            
            Return as JSON with extracted parameters.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=extraction_prompt,
                task_type="analytical",
                context={"parameter_extraction": True}
            )
            
            try:
                params = json.loads(response.get("text", "{}"))
                
                # Add defaults
                if media_type == "image":
                    params.setdefault("style", "photorealistic")
                    params.setdefault("size", "1024x1024")
                else:  # video
                    params.setdefault("style", "cinematic")
                    params.setdefault("duration", 5)
                
                params.setdefault("prompt", message)
                
                return params
                
            except json.JSONDecodeError:
                return {
                    "prompt": message,
                    "style": "photorealistic" if media_type == "image" else "cinematic"
                }
                
        except Exception as e:
            logger.error(f"Parameter extraction failed: {str(e)}")
            return {"prompt": message}
    
    async def _enhance_image_prompt(self, prompt: str, style: str) -> str:
        """Enhance image prompt with style and quality modifiers"""
        try:
            style_modifier = self.image_styles.get(style, "")
            
            enhancement_prompt = f"""
            Enhance this image generation prompt for better results:
            
            Original: "{prompt}"
            Style: {style}
            
            Create a detailed, specific prompt that includes:
            1. Clear subject description
            2. Visual style and aesthetics
            3. Lighting and composition
            4. Quality modifiers
            5. Technical specifications
            
            Keep it concise but descriptive. Return only the enhanced prompt.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=enhancement_prompt,
                task_type="creative",
                context={"prompt_enhancement": True}
            )
            
            enhanced = response.get("text", prompt).strip()
            
            # Add style modifier
            if style_modifier:
                enhanced = f"{enhanced}, {style_modifier}"
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}")
            return f"{prompt}, {self.image_styles.get(style, '')}"
    
    async def _enhance_video_prompt(self, prompt: str, style: str) -> str:
        """Enhance video prompt with style and quality modifiers"""
        try:
            style_modifier = self.video_styles.get(style, "")
            
            enhancement_prompt = f"""
            Enhance this video generation prompt for better results:
            
            Original: "{prompt}"
            Style: {style}
            
            Create a detailed prompt that includes:
            1. Clear scene description
            2. Camera movement and angles
            3. Visual style and mood
            4. Motion and action
            5. Quality specifications
            
            Keep it concise but descriptive. Return only the enhanced prompt.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=enhancement_prompt,
                task_type="creative",
                context={"prompt_enhancement": True}
            )
            
            enhanced = response.get("text", prompt).strip()
            
            # Add style modifier
            if style_modifier:
                enhanced = f"{enhanced}, {style_modifier}"
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Video prompt enhancement failed: {str(e)}")
            return f"{prompt}, {self.video_styles.get(style, '')}"
    
    async def _generate_image_with_provider(
        self,
        provider_name: str,
        provider_config: Dict[str, Any],
        prompt: str,
        size: str
    ) -> Dict[str, Any]:
        """Generate image with specific provider"""
        try:
            if provider_name == "huggingface":
                return await self._generate_image_huggingface(provider_config, prompt)
            elif provider_name == "replicate":
                return await self._generate_image_replicate(provider_config, prompt, size)
            elif provider_name == "automatic1111":
                return await self._generate_image_automatic1111(provider_config, prompt, size)
            else:
                return {"success": False, "error": f"Unknown provider: {provider_name}"}
                
        except Exception as e:
            logger.error(f"Image generation with {provider_name} failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_image_huggingface(
        self,
        config: Dict[str, Any],
        prompt: str
    ) -> Dict[str, Any]:
        """Generate image using Hugging Face API"""
        try:
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {"inputs": prompt}
            
            async with self.session.post(
                config["endpoint"],
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return {
                        "success": True,
                        "image_data": image_data,
                        "format": "png"
                    }
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Hugging Face error: {error_text}"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_image_replicate(
        self,
        config: Dict[str, Any],
        prompt: str,
        size: str
    ) -> Dict[str, Any]:
        """Generate image using Replicate API"""
        try:
            headers = {
                "Authorization": f"Token {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            width, height = size.split("x")
            
            payload = {
                "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                "input": {
                    "prompt": prompt,
                    "width": int(width),
                    "height": int(height),
                    "num_outputs": 1,
                    "scheduler": "K_EULER",
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5
                }
            }
            
            # Create prediction
            async with self.session.post(
                config["endpoint"],
                headers=headers,
                json=payload
            ) as response:
                if response.status == 201:
                    prediction = await response.json()
                    prediction_id = prediction["id"]
                    
                    # Poll for completion
                    for _ in range(60):  # Wait up to 5 minutes
                        await asyncio.sleep(5)
                        
                        async with self.session.get(
                            f"{config['endpoint']}/{prediction_id}",
                            headers=headers
                        ) as status_response:
                            if status_response.status == 200:
                                status_data = await status_response.json()
                                
                                if status_data["status"] == "succeeded":
                                    image_url = status_data["output"][0]
                                    
                                    # Download image
                                    async with self.session.get(image_url) as img_response:
                                        if img_response.status == 200:
                                            image_data = await img_response.read()
                                            return {
                                                "success": True,
                                                "image_data": image_data,
                                                "format": "png"
                                            }
                                
                                elif status_data["status"] == "failed":
                                    return {"success": False, "error": "Generation failed"}
                    
                    return {"success": False, "error": "Generation timeout"}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Replicate error: {error_text}"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_image_automatic1111(
        self,
        config: Dict[str, Any],
        prompt: str,
        size: str
    ) -> Dict[str, Any]:
        """Generate image using Automatic1111 local installation"""
        try:
            width, height = map(int, size.split("x"))
            
            payload = {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, ugly",
                "width": width,
                "height": height,
                "steps": 20,
                "cfg_scale": 7,
                "sampler_name": "Euler a",
                "batch_size": 1,
                "n_iter": 1
            }
            
            async with self.session.post(
                config["endpoint"],
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("images"):
                        # Decode base64 image
                        image_b64 = data["images"][0]
                        image_data = base64.b64decode(image_b64)
                        
                        return {
                            "success": True,
                            "image_data": image_data,
                            "format": "png"
                        }
                    else:
                        return {"success": False, "error": "No image generated"}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Automatic1111 error: {error_text}"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_video_with_provider(
        self,
        provider_name: str,
        provider_config: Dict[str, Any],
        prompt: str,
        duration: int
    ) -> Dict[str, Any]:
        """Generate video with specific provider"""
        try:
            if provider_name == "replicate_video":
                return await self._generate_video_replicate(provider_config, prompt, duration)
            else:
                return {"success": False, "error": f"Unknown video provider: {provider_name}"}
                
        except Exception as e:
            logger.error(f"Video generation with {provider_name} failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_video_replicate(
        self,
        config: Dict[str, Any],
        prompt: str,
        duration: int
    ) -> Dict[str, Any]:
        """Generate video using Replicate API"""
        try:
            headers = {
                "Authorization": f"Token {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "version": "9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
                "input": {
                    "prompt": prompt,
                    "num_frames": duration * 8,  # 8 fps
                    "num_inference_steps": 20
                }
            }
            
            # Create prediction
            async with self.session.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 201:
                    prediction = await response.json()
                    prediction_id = prediction["id"]
                    
                    # Poll for completion (videos take longer)
                    for _ in range(120):  # Wait up to 10 minutes
                        await asyncio.sleep(5)
                        
                        async with self.session.get(
                            f"https://api.replicate.com/v1/predictions/{prediction_id}",
                            headers=headers
                        ) as status_response:
                            if status_response.status == 200:
                                status_data = await status_response.json()
                                
                                if status_data["status"] == "succeeded":
                                    video_url = status_data["output"]
                                    
                                    # Download video
                                    async with self.session.get(video_url) as vid_response:
                                        if vid_response.status == 200:
                                            video_data = await vid_response.read()
                                            return {
                                                "success": True,
                                                "video_data": video_data,
                                                "format": "mp4"
                                            }
                                
                                elif status_data["status"] == "failed":
                                    return {"success": False, "error": "Video generation failed"}
                    
                    return {"success": False, "error": "Video generation timeout"}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"Replicate error: {error_text}"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _save_generated_image(
        self,
        image_data: bytes,
        user_id: str,
        provider: str
    ) -> str:
        """Save generated image to file system"""
        try:
            timestamp = int(time.time())
            filename = f"image_{user_id}_{provider}_{timestamp}.png"
            file_path = Path(self.settings.GENERATED_MEDIA_DIR) / filename
            
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Image saving failed: {str(e)}")
            raise
    
    async def _save_generated_video(
        self,
        video_data: bytes,
        user_id: str,
        provider: str
    ) -> str:
        """Save generated video to file system"""
        try:
            timestamp = int(time.time())
            filename = f"video_{user_id}_{provider}_{timestamp}.mp4"
            file_path = Path(self.settings.GENERATED_MEDIA_DIR) / filename
            
            with open(file_path, 'wb') as f:
                f.write(video_data)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Video saving failed: {str(e)}")
            raise
    
    async def _test_providers(self):
        """Test availability of media generation providers"""
        logger.info("Testing media generation provider availability...")
        
        # Test image providers
        for name, config in self.image_providers.items():
            try:
                if config["enabled"]:
                    if config.get("local"):
                        # Test local provider
                        try:
                            async with self.session.get(f"{config['endpoint'].replace('/sdapi/v1/txt2img', '')}/") as response:
                                available = response.status == 200
                        except:
                            available = False
                    else:
                        # Remote provider - assume available if API key exists
                        available = bool(config.get("api_key"))
                    
                    status = "✅ Available" if available else "❌ Unavailable"
                    logger.info(f"  Image - {name}: {status}")
                    config["enabled"] = available
                else:
                    logger.info(f"  Image - {name}: ❌ Disabled (no API key)")
                    
            except Exception as e:
                logger.warning(f"  Image - {name}: ❌ Test failed - {str(e)}")
                config["enabled"] = False
        
        # Test video providers
        for name, config in self.video_providers.items():
            try:
                if config["enabled"]:
                    available = bool(config.get("api_key"))
                    status = "✅ Available" if available else "❌ Unavailable"
                    logger.info(f"  Video - {name}: {status}")
                    config["enabled"] = available
                else:
                    logger.info(f"  Video - {name}: ❌ Disabled (no API key)")
                    
            except Exception as e:
                logger.warning(f"  Video - {name}: ❌ Test failed - {str(e)}")
                config["enabled"] = False
    
    def _generate_cache_key(
        self,
        prompt: str,
        style: str,
        size_or_duration: str,
        media_type: str
    ) -> str:
        """Generate cache key for media generation"""
        key_data = f"{prompt}:{style}:{size_or_duration}:{media_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached generation result"""
        if cache_key in self.generation_cache:
            cached_data, timestamp = self.generation_cache[cache_key]
            if time.time() - timestamp < 3600:  # 1 hour cache
                cached_data["cached"] = True
                return cached_data
            else:
                del self.generation_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache generation result"""
        self.generation_cache[cache_key] = (result.copy(), time.time())
        
        # Limit cache size
        if len(self.generation_cache) > 100:
            oldest_key = min(
                self.generation_cache.keys(),
                key=lambda k: self.generation_cache[k][1]
            )
            del self.generation_cache[oldest_key]
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for media generation service"""
        enabled_image_providers = sum(
            1 for config in self.image_providers.values()
            if config["enabled"]
        )
        
        enabled_video_providers = sum(
            1 for config in self.video_providers.values()
            if config["enabled"]
        )
        
        return {
            "status": "healthy" if (enabled_image_providers > 0 or enabled_video_providers > 0) else "degraded",
            "image_providers": {
                "enabled": enabled_image_providers,
                "total": len(self.image_providers)
            },
            "video_providers": {
                "enabled": enabled_video_providers,
                "total": len(self.video_providers)
            },
            "cache_size": len(self.generation_cache),
            "media_directory": str(Path(self.settings.GENERATED_MEDIA_DIR).absolute()),
            "session_active": self.session is not None and not self.session.closed
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
