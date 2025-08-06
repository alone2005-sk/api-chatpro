import asyncio
import io
import os
import tempfile
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import base64
import json

# Text-to-Speech
import pyttsx3
from gtts import gTTS

# Speech-to-Text
import speech_recognition as sr

# Audio processing
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import librosa
import soundfile as sf
import numpy as np

# Voice cloning and synthesis (placeholder for advanced models)
# import torch
# from TTS.api import TTS

from core.logger import get_logger
from core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class VoiceService:
    """Advanced voice processing service for TTS, STT, and voice synthesis"""
    async def initialize(self):
        """
        Optional startup logic. Add any preloading or directory scanning here.
        """
        pass
    def __init__(self, settings = None):
        self.settings = settings
        self.tts_engines = {
            'pyttsx3': self._tts_pyttsx3,
            'gtts': self._tts_gtts,
            'elevenlabs': self._tts_elevenlabs,  # Requires API key
            'azure': self._tts_azure,  # Requires API key
            'aws_polly': self._tts_aws_polly  # Requires API key
        }
        
        self.stt_engines = {
            'google': self._stt_google,
            'azure': self._stt_azure,
            'aws_transcribe': self._stt_aws_transcribe,
            'whisper': self._stt_whisper
        }
        
        self.supported_formats = ['wav', 'mp3', 'flac', 'ogg', 'm4a']
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize TTS engine
        try:
            self.pyttsx3_engine = pyttsx3.init()
            self._configure_pyttsx3()
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3: {str(e)}")
            self.pyttsx3_engine = None
    
    def _configure_pyttsx3(self):
        """Configure pyttsx3 TTS engine"""
        if self.pyttsx3_engine:
            # Set properties
            self.pyttsx3_engine.setProperty('rate', 150)  # Speed
            self.pyttsx3_engine.setProperty('volume', 0.9)  # Volume
            
            # Get available voices
            voices = self.pyttsx3_engine.getProperty('voices')
            if voices:
                # Use first available voice
                self.pyttsx3_engine.setProperty('voice', voices[0].id)
    
    async def text_to_speech(self, text: str, voice_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert text to speech with various options"""
        try:
            if not text or not text.strip():
                return {'error': 'No text provided for TTS'}
            
            # Default configuration
            config = {
                'engine': 'gtts',
                'language': 'en',
                'speed': 'normal',
                'voice': 'default',
                'format': 'mp3',
                'quality': 'high'
            }
            
            if voice_config:
                config.update(voice_config)
            
            # Generate speech
            engine = config.get('engine', 'gtts')
            if engine in self.tts_engines:
                audio_data = await self.tts_engines[engine](text, config)
            else:
                return {'error': f'Unsupported TTS engine: {engine}'}
            
            if 'error' in audio_data:
                return audio_data
            
            # Process audio if needed
            if config.get('effects'):
                audio_data = await self._apply_audio_effects(audio_data, config['effects'])
            
            return {
                'success': True,
                'audio_data': audio_data['audio_data'],
                'format': audio_data['format'],
                'duration': audio_data.get('duration', 0),
                'text': text,
                'config': config,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")
            return {'error': f'TTS failed: {str(e)}'}
    
    async def speech_to_text(self, audio_data: bytes, audio_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert speech to text with various options"""
        try:
            if not audio_data:
                return {'error': 'No audio data provided for STT'}
            
            # Default configuration
            config = {
                'engine': 'google',
                'language': 'en-US',
                'format': 'auto',
                'enhance_audio': True,
                'confidence_threshold': 0.5
            }
            
            if audio_config:
                config.update(audio_config)
            
            # Preprocess audio if needed
            if config.get('enhance_audio'):
                audio_data = await self._enhance_audio(audio_data)
            
            # Convert to text
            engine = config.get('engine', 'google')
            if engine in self.stt_engines:
                text_result = await self.stt_engines[engine](audio_data, config)
            else:
                return {'error': f'Unsupported STT engine: {engine}'}
            
            if 'error' in text_result:
                return text_result
            
            return {
                'success': True,
                'text': text_result['text'],
                'confidence': text_result.get('confidence', 0),
                'language': text_result.get('language', config['language']),
                'config': config,
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"STT failed: {str(e)}")
            return {'error': f'STT failed: {str(e)}'}
    
    async def _tts_pyttsx3(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Text-to-speech using pyttsx3 (offline)"""
        try:
            if not self.pyttsx3_engine:
                return {'error': 'pyttsx3 engine not available'}
            
            # Configure voice settings
            rate = {'slow': 120, 'normal': 150, 'fast': 200}.get(config.get('speed', 'normal'), 150)
            self.pyttsx3_engine.setProperty('rate', rate)
            
            # Generate audio file
            temp_file = os.path.join(self.temp_dir, f"tts_{datetime.now().timestamp()}.wav")
            self.pyttsx3_engine.save_to_file(text, temp_file)
            self.pyttsx3_engine.runAndWait()
            
            # Read audio data
            with open(temp_file, 'rb') as f:
                audio_data = f.read()
            
            # Get duration
            audio = AudioSegment.from_wav(temp_file)
            duration = len(audio) / 1000.0
            
            # Cleanup
            os.remove(temp_file)
            
            return {
                'audio_data': base64.b64encode(audio_data).decode(),
                'format': 'wav',
                'duration': duration
            }
            
        except Exception as e:
            return {'error': f'pyttsx3 TTS failed: {str(e)}'}
    
    async def _tts_gtts(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Text-to-speech using Google TTS"""
        try:
            language = config.get('language', 'en')
            
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=config.get('speed') == 'slow')
            
            # Save to temporary file
            temp_file = os.path.join(self.temp_dir, f"gtts_{datetime.now().timestamp()}.mp3")
            tts.save(temp_file)
            
            # Read audio data
            with open(temp_file, 'rb') as f:
                audio_data = f.read()
            
            # Get duration
            audio = AudioSegment.from_mp3(temp_file)
            duration = len(audio) / 1000.0
            
            # Cleanup
            os.remove(temp_file)
            
            return {
                'audio_data': base64.b64encode(audio_data).decode(),
                'format': 'mp3',
                'duration': duration
            }
            
        except Exception as e:
            return {'error': f'gTTS failed: {str(e)}'}
    
    async def _tts_elevenlabs(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Text-to-speech using ElevenLabs API"""
        try:
            if not hasattr(settings, 'ELEVENLABS_API_KEY'):
                return {'error': 'ElevenLabs API key not configured'}
            
            import aiohttp
            
            url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": settings.ELEVENLABS_API_KEY
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        # Get duration
                        temp_file = os.path.join(self.temp_dir, f"elevenlabs_{datetime.now().timestamp()}.mp3")
                        with open(temp_file, 'wb') as f:
                            f.write(audio_data)
                        
                        audio = AudioSegment.from_mp3(temp_file)
                        duration = len(audio) / 1000.0
                        os.remove(temp_file)
                        
                        return {
                            'audio_data': base64.b64encode(audio_data).decode(),
                            'format': 'mp3',
                            'duration': duration
                        }
                    else:
                        return {'error': f'ElevenLabs API error: {response.status}'}
                        
        except Exception as e:
            return {'error': f'ElevenLabs TTS failed: {str(e)}'}
    
    async def _tts_azure(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Text-to-speech using Azure Cognitive Services"""
        try:
            if not hasattr(settings, 'AZURE_SPEECH_KEY'):
                return {'error': 'Azure Speech API key not configured'}
            
            import aiohttp
            
            region = getattr(settings, 'AZURE_SPEECH_REGION', 'eastus')
            url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
            
            headers = {
                'Ocp-Apim-Subscription-Key': settings.AZURE_SPEECH_KEY,
                'Content-Type': 'application/ssml+xml',
                'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
            }
            
            voice_name = config.get('voice', 'en-US-JennyNeural')
            ssml = f"""
            <speak version='1.0' xml:lang='en-US'>
                <voice xml:lang='en-US' xml:gender='Female' name='{voice_name}'>
                    {text}
                </voice>
            </speak>
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=ssml, headers=headers) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        # Get duration
                        temp_file = os.path.join(self.temp_dir, f"azure_{datetime.now().timestamp()}.mp3")
                        with open(temp_file, 'wb') as f:
                            f.write(audio_data)
                        
                        audio = AudioSegment.from_mp3(temp_file)
                        duration = len(audio) / 1000.0
                        os.remove(temp_file)
                        
                        return {
                            'audio_data': base64.b64encode(audio_data).decode(),
                            'format': 'mp3',
                            'duration': duration
                        }
                    else:
                        return {'error': f'Azure TTS API error: {response.status}'}
                        
        except Exception as e:
            return {'error': f'Azure TTS failed: {str(e)}'}
    
    async def _tts_aws_polly(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Text-to-speech using AWS Polly"""
        try:
            try:
                import boto3
            except ImportError:
                return {'error': 'boto3 not installed for AWS Polly'}
            
            if not hasattr(settings, 'AWS_ACCESS_KEY_ID'):
                return {'error': 'AWS credentials not configured'}
            
            polly = boto3.client(
                'polly',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=getattr(settings, 'AWS_REGION', 'us-east-1')
            )
            
            response = polly.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=config.get('voice', 'Joanna'),
                Engine='neural'
            )
            
            audio_data = response['AudioStream'].read()
            
            # Get duration
            temp_file = os.path.join(self.temp_dir, f"polly_{datetime.now().timestamp()}.mp3")
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            audio = AudioSegment.from_mp3(temp_file)
            duration = len(audio) / 1000.0
            os.remove(temp_file)
            
            return {
                'audio_data': base64.b64encode(audio_data).decode(),
                'format': 'mp3',
                'duration': duration
            }
            
        except Exception as e:
            return {'error': f'AWS Polly TTS failed: {str(e)}'}
    
    async def _stt_google(self, audio_data: bytes, config: Dict[str, Any]) -> Dict[str, Any]:
        """Speech-to-text using Google Speech Recognition"""
        try:
            # Save audio to temporary file
            temp_file = os.path.join(self.temp_dir, f"stt_{datetime.now().timestamp()}.wav")
            
            # Convert to WAV format if needed
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            audio_segment.export(temp_file, format="wav")
            
            # Use speech recognition
            with sr.AudioFile(temp_file) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech
            text = self.recognizer.recognize_google(
                audio, 
                language=config.get('language', 'en-US')
            )
            
            # Cleanup
            os.remove(temp_file)
            
            return {
                'text': text,
                'confidence': 0.8,  # Google doesn't provide confidence scores in free tier
                'language': config.get('language', 'en-US')
            }
            
        except sr.UnknownValueError:
            return {'error': 'Could not understand audio'}
        except sr.RequestError as e:
            return {'error': f'Google STT service error: {str(e)}'}
        except Exception as e:
            return {'error': f'Google STT failed: {str(e)}'}
    
    async def _stt_azure(self, audio_data: bytes, config: Dict[str, Any]) -> Dict[str, Any]:
        """Speech-to-text using Azure Cognitive Services"""
        try:
            if not hasattr(settings, 'AZURE_SPEECH_KEY'):
                return {'error': 'Azure Speech API key not configured'}
            
            import aiohttp
            
            region = getattr(settings, 'AZURE_SPEECH_REGION', 'eastus')
            url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            
            headers = {
                'Ocp-Apim-Subscription-Key': settings.AZURE_SPEECH_KEY,
                'Content-Type': 'audio/wav'
            }
            
            params = {
                'language': config.get('language', 'en-US'),
                'format': 'detailed'
            }
            
            # Convert to WAV format
            temp_file = os.path.join(self.temp_dir, f"azure_stt_{datetime.now().timestamp()}.wav")
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            audio_segment.export(temp_file, format="wav")
            
            with open(temp_file, 'rb') as f:
                wav_data = f.read()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=wav_data, headers=headers, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get('RecognitionStatus') == 'Success':
                            best_result = result.get('NBest', [{}])[0]
                            
                            os.remove(temp_file)
                            
                            return {
                                'text': best_result.get('Display', ''),
                                'confidence': best_result.get('Confidence', 0),
                                'language': config.get('language', 'en-US')
                            }
                        else:
                            os.remove(temp_file)
                            return {'error': f'Azure STT recognition failed: {result.get("RecognitionStatus")}'}
                    else:
                        os.remove(temp_file)
                        return {'error': f'Azure STT API error: {response.status}'}
                        
        except Exception as e:
            return {'error': f'Azure STT failed: {str(e)}'}
    
    async def _stt_aws_transcribe(self, audio_data: bytes, config: Dict[str, Any]) -> Dict[str, Any]:
        """Speech-to-text using AWS Transcribe"""
        try:
            try:
                import boto3
            except ImportError:
                return {'error': 'boto3 not installed for AWS Transcribe'}
            
            if not hasattr(settings, 'AWS_ACCESS_KEY_ID'):
                return {'error': 'AWS credentials not configured'}
            
            # AWS Transcribe requires files to be uploaded to S3 first
            # This is a simplified implementation
            return {'error': 'AWS Transcribe requires S3 integration (not implemented in this demo)'}
            
        except Exception as e:
            return {'error': f'AWS Transcribe failed: {str(e)}'}
    
    async def _stt_whisper(self, audio_data: bytes, config: Dict[str, Any]) -> Dict[str, Any]:
        """Speech-to-text using OpenAI Whisper"""
        try:
            try:
                import whisper
            except ImportError:
                return {'error': 'whisper not installed'}
            
            # Load Whisper model
            model_size = config.get('model_size', 'base')
            model = whisper.load_model(model_size)
            
            # Save audio to temporary file
            temp_file = os.path.join(self.temp_dir, f"whisper_{datetime.now().timestamp()}.wav")
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            audio_segment.export(temp_file, format="wav")
            
            # Transcribe
            result = model.transcribe(temp_file)
            
            # Cleanup
            os.remove(temp_file)
            
            return {
                'text': result['text'].strip(),
                'confidence': 0.9,  # Whisper doesn't provide confidence scores
                'language': result.get('language', config.get('language', 'en'))
            }
            
        except Exception as e:
            return {'error': f'Whisper STT failed: {str(e)}'}
    
    async def _enhance_audio(self, audio_data: bytes) -> bytes:
        """Enhance audio quality for better STT results"""
        try:
            # Load audio
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Normalize audio
            audio = normalize(audio)
            
            # Apply dynamic range compression
            audio = compress_dynamic_range(audio)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate to 16kHz (optimal for most STT engines)
            audio = audio.set_frame_rate(16000)
            
            # Export enhanced audio
            enhanced_buffer = io.BytesIO()
            audio.export(enhanced_buffer, format="wav")
            
            return enhanced_buffer.getvalue()
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {str(e)}")
            return audio_data  # Return original if enhancement fails
    
    async def _apply_audio_effects(self, audio_data: Dict[str, Any], effects: List[str]) -> Dict[str, Any]:
        """Apply audio effects to generated speech"""
        try:
            # Decode audio data
            audio_bytes = base64.b64decode(audio_data['audio_data'])
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_data['format'])
            
            # Apply effects
            for effect in effects:
                if effect == 'reverb':
                    # Simple reverb simulation
                    reverb = audio.overlay(audio - 10, position=100)
                    audio = audio.overlay(reverb - 15, position=200)
                elif effect == 'echo':
                    # Add echo effect
                    echo = audio - 10
                    audio = audio.overlay(echo, position=500)
                elif effect == 'speed_up':
                    # Increase playback speed
                    audio = audio.speedup(playback_speed=1.2)
                elif effect == 'slow_down':
                    # Decrease playback speed
                    audio = audio.speedup(playback_speed=0.8)
                elif effect == 'pitch_up':
                    # Increase pitch (simplified)
                    audio = audio._spawn(audio.raw_data, overrides={'frame_rate': int(audio.frame_rate * 1.2)})
                elif effect == 'pitch_down':
                    # Decrease pitch (simplified)
                    audio = audio._spawn(audio.raw_data, overrides={'frame_rate': int(audio.frame_rate * 0.8)})
            
            # Export processed audio
            processed_buffer = io.BytesIO()
            audio.export(processed_buffer, format=audio_data['format'])
            
            return {
                'audio_data': base64.b64encode(processed_buffer.getvalue()).decode(),
                'format': audio_data['format'],
                'duration': len(audio) / 1000.0
            }
            
        except Exception as e:
            logger.warning(f"Audio effects failed: {str(e)}")
            return audio_data  # Return original if effects fail
    
    async def analyze_voice(self, audio_data: bytes) -> Dict[str, Any]:
        """Analyze voice characteristics from audio"""
        try:
            # Save audio to temporary file
            temp_file = os.path.join(self.temp_dir, f"analyze_{datetime.now().timestamp()}.wav")
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            audio_segment.export(temp_file, format="wav")
            
            # Load with librosa for analysis
            y, sr = librosa.load(temp_file)
            
            # Extract features
            features = {
                'duration': len(audio_segment) / 1000.0,
                'sample_rate': sr,
                'channels': audio_segment.channels,
                'frame_rate': audio_segment.frame_rate,
                'loudness_db': audio_segment.dBFS,
                
                # Voice characteristics
                'fundamental_frequency': float(librosa.yin(y, fmin=50, fmax=400).mean()),
                'spectral_centroid': float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
                'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(y).mean()),
                'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                
                # MFCC features (voice fingerprint)
                'mfcc_features': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
                
                # Energy and dynamics
                'rms_energy': float(librosa.feature.rms(y=y).mean()),
                'spectral_rolloff': float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean()),
                'spectral_bandwidth': float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
            }
            
            # Voice quality assessment
            features['voice_quality'] = {
                'clarity': self._assess_clarity(features),
                'pitch_stability': self._assess_pitch_stability(y, sr),
                'volume_consistency': self._assess_volume_consistency(y)
            }
            
            # Cleanup
            os.remove(temp_file)
            
            return {
                'success': True,
                'features': features,
                'analyzed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Voice analysis failed: {str(e)}")
            return {'error': f'Voice analysis failed: {str(e)}'}
    
    def _assess_clarity(self, features: Dict[str, Any]) -> float:
        """Assess voice clarity based on spectral features"""
        # Simple clarity assessment based on spectral centroid and bandwidth
        centroid = features.get('spectral_centroid', 0)
        bandwidth = features.get('spectral_bandwidth', 0)
        
        # Normalize and combine features
        clarity_score = min(1.0, (centroid / 3000) * 0.6 + (bandwidth / 2000) * 0.4)
        return round(clarity_score, 3)
    
    def _assess_pitch_stability(self, y: np.ndarray, sr: int) -> float:
        """Assess pitch stability"""
        try:
            f0 = librosa.yin(y, fmin=50, fmax=400)
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames
            
            if len(f0_clean) > 0:
                stability = 1.0 - (np.std(f0_clean) / np.mean(f0_clean))
                return round(max(0, min(1, stability)), 3)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _assess_volume_consistency(self, y: np.ndarray) -> float:
        """Assess volume consistency"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            if len(rms) > 0:
                consistency = 1.0 - (np.std(rms) / np.mean(rms))
                return round(max(0, min(1, consistency)), 3)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    async def get_available_voices(self, engine: str = 'all') -> Dict[str, Any]:
        """Get available voices for TTS engines"""
        voices = {}
        
        if engine in ['all', 'pyttsx3'] and self.pyttsx3_engine:
            try:
                pyttsx3_voices = self.pyttsx3_engine.getProperty('voices')
                voices['pyttsx3'] = [
                    {
                        'id': voice.id,
                        'name': voice.name,
                        'languages': getattr(voice, 'languages', []),
                        'gender': getattr(voice, 'gender', 'unknown')
                    }
                    for voice in pyttsx3_voices
                ]
            except Exception as e:
                logger.warning(f"Failed to get pyttsx3 voices: {str(e)}")
        
        if engine in ['all', 'gtts']:
            voices['gtts'] = {
                'supported_languages': [
                    'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
                    'ar', 'hi', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi'
                ]
            }
        
        if engine in ['all', 'azure']:
            voices['azure'] = {
                'note': 'Azure provides 200+ voices across 60+ languages',
                'popular_voices': [
                    'en-US-JennyNeural', 'en-US-GuyNeural', 'en-GB-SoniaNeural',
                    'es-ES-ElviraNeural', 'fr-FR-DeniseNeural', 'de-DE-KatjaNeural'
                ]
            }
        
        return {
            'available_voices': voices,
            'engines': list(self.tts_engines.keys()),
            'retrieved_at': datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for voice service"""
        return {
            'status': 'healthy',
            'tts_engines': list(self.tts_engines.keys()),
            'stt_engines': list(self.stt_engines.keys()),
            'supported_formats': self.supported_formats,
            'pyttsx3_available': self.pyttsx3_engine is not None,
            'temp_dir': self.temp_dir
        }
