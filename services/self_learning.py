"""
Self-Learning Service - Background learning from user interactions
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import hashlib
import re

from core.config import get_settings
from core.database import DatabaseManager
from core.logger import get_logger
from services.llm_orchestrator import LLMOrchestrator

logger = get_logger(__name__)

class SelfLearningService:
    """Background service for continuous learning from user interactions"""
    
    def __init__(self, llm_orchestrator: LLMOrchestrator, db_manager: DatabaseManager):
        self.settings = get_settings()
        self.llm_orchestrator = llm_orchestrator
        self.db_manager = db_manager
        
        # Learning configuration
        self.learning_config = {
            "batch_size": self.settings.LEARNING_BATCH_SIZE,
            "learning_interval": self.settings.LEARNING_INTERVAL,
            "min_confidence_threshold": 0.7,
            "pattern_similarity_threshold": 0.8,
            "quality_improvement_threshold": 0.1,
            "max_learning_history": 10000
        }
        
        # Pattern extractors
        self.pattern_extractors = {
            "intent": self._extract_intent_patterns,
            "topic": self._extract_topic_patterns,
            "complexity": self._extract_complexity_patterns,
            "style": self._extract_style_patterns,
            "sentiment": self._extract_sentiment_patterns,
            "context": self._extract_context_patterns
        }
        
        # Knowledge base (in-memory for fast access)
        self.knowledge_base = {
            "patterns": defaultdict(dict),
            "user_preferences": defaultdict(dict),
            "response_templates": defaultdict(list),
            "improvement_suggestions": defaultdict(list),
            "quality_scores": defaultdict(float)
        }
        
        # Learning statistics
        self.learning_stats = {
            "total_interactions_processed": 0,
            "patterns_discovered": 0,
            "quality_improvements": 0,
            "learning_sessions": 0,
            "last_learning_session": None,
            "average_confidence": 0.0,
            "knowledge_base_size": 0
        }
        
        # Background task
        self.learning_task = None
        self.is_running = False
    
    async def initialize(self):
        """Initialize the self-learning service"""
        try:
            logger.info("Initializing Self-Learning Service...")
            
            # Load existing knowledge base
            await self._load_knowledge_base()
            
            # Start background learning task
            if self.settings.ENABLE_SELF_LEARNING:
                self.learning_task = asyncio.create_task(self._background_learning_loop())
                self.is_running = True
            
            logger.info("✅ Self-Learning Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Self-Learning Service: {str(e)}")
            raise
    
    async def learn_from_interaction(
        self,
        user_id: str,
        chat_id: str,
        user_input: str,
        ai_response: str,
        user_feedback: Optional[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ):
        """Learn from a single user interaction"""
        try:
            # Calculate quality score
            quality_score = self._calculate_quality_score(ai_response, user_feedback, context)
            
            # Extract patterns
            patterns = await self._extract_all_patterns(user_input, context)
            
            # Save to database for batch processing
            await self.db_manager.save_learning_interaction(
                user_id=user_id,
                chat_id=chat_id,
                user_input=user_input,
                ai_response=ai_response,
                user_feedback=user_feedback,
                context=context,
                patterns=patterns,
                quality_score=quality_score
            )
            
            # Update real-time knowledge if high quality
            if quality_score > 0.8:
                await self._update_knowledge_immediately(
                    user_id, patterns, ai_response, quality_score
                )
            
        except Exception as e:
            logger.error(f"Learning from interaction failed: {str(e)}")
    
    async def generate_improved_response(
        self,
        user_input: str,
        context: Dict[str, Any] = None,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """Generate improved response using learned knowledge"""
        try:
            # Extract patterns from input
            patterns = await self._extract_all_patterns(user_input, context)
            
            # Find similar patterns in knowledge base
            similar_patterns = self._find_similar_patterns(patterns, user_id)
            
            # Calculate confidence
            confidence = self._calculate_response_confidence(similar_patterns, patterns)
            
            if confidence < self.learning_config["min_confidence_threshold"]:
                return {
                    "success": False,
                    "confidence": confidence,
                    "reason": "Insufficient confidence for learned response"
                }
            
            # Generate improved response
            improved_response = await self._generate_learned_response(
                user_input, patterns, similar_patterns, context, user_id
            )
            
            return {
                "success": True,
                "response": improved_response,
                "confidence": confidence,
                "patterns_used": len(similar_patterns),
                "learning_source": "self_learning"
            }
            
        except Exception as e:
            logger.error(f"Improved response generation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _background_learning_loop(self):
        """Background learning loop"""
        logger.info("Starting background learning loop...")
        
        while self.is_running:
            try:
                # Process unprocessed interactions
                await self._process_learning_batch()
                
                # Optimize knowledge base periodically
                if self.learning_stats["learning_sessions"] % 10 == 0:
                    await self._optimize_knowledge_base()
                
                # Save knowledge base periodically
                if self.learning_stats["learning_sessions"] % 5 == 0:
                    await self._save_knowledge_base()
                
                # Wait for next iteration
                await asyncio.sleep(self.learning_config["learning_interval"])
                
            except Exception as e:
                logger.error(f"Background learning error: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _process_learning_batch(self):
        """Process a batch of learning interactions"""
        try:
            # Get unprocessed interactions
            interactions = await self.db_manager.get_unprocessed_learning_interactions(
                limit=self.learning_config["batch_size"]
            )
            
            if not interactions:
                return
            
            logger.info(f"Processing {len(interactions)} learning interactions...")
            
            # Process each interaction
            processed_ids = []
            for interaction in interactions:
                try:
                    await self._process_single_interaction(interaction)
                    processed_ids.append(interaction.id)
                except Exception as e:
                    logger.error(f"Failed to process interaction {interaction.id}: {str(e)}")
            
            # Mark as processed
            if processed_ids:
                await self.db_manager.mark_learning_interactions_processed(processed_ids)
                
                self.learning_stats["total_interactions_processed"] += len(processed_ids)
                self.learning_stats["learning_sessions"] += 1
                self.learning_stats["last_learning_session"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Learning batch processing failed: {str(e)}")
    
    async def _process_single_interaction(self, interaction):
        """Process a single learning interaction"""
        try:
            patterns = interaction.patterns or {}
            quality_score = interaction.quality_score
            user_id = interaction.user_id
            
            # Update pattern knowledge
            await self._update_pattern_knowledge(patterns, quality_score)
            
            # Update user preferences
            await self._update_user_preferences(user_id, patterns, quality_score)
            
            # Update response templates
            await self._update_response_templates(
                interaction.user_input,
                interaction.ai_response,
                patterns,
                quality_score
            )
            
            # Generate improvement suggestions
            if quality_score < 0.6:  # Low quality response
                suggestions = await self._generate_improvement_suggestions(
                    interaction.user_input,
                    interaction.ai_response,
                    patterns
                )
                self.knowledge_base["improvement_suggestions"][user_id].extend(suggestions)
            
        except Exception as e:
            logger.error(f"Single interaction processing failed: {str(e)}")
    
    async def _extract_all_patterns(
        self,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract all types of patterns from user input"""
        patterns = {}
        
        for pattern_type, extractor in self.pattern_extractors.items():
            try:
                patterns[pattern_type] = await extractor(user_input, context)
            except Exception as e:
                logger.error(f"Pattern extraction failed for {pattern_type}: {str(e)}")
                patterns[pattern_type] = {}
        
        return patterns
    
    async def _extract_intent_patterns(
        self,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract intent patterns"""
        intent_keywords = {
            "question": ["what", "how", "why", "when", "where", "who", "?"],
            "request": ["please", "can you", "could you", "would you", "help me"],
            "command": ["create", "make", "build", "generate", "do", "run", "execute"],
            "information": ["tell me", "explain", "describe", "show me", "what is"],
            "problem": ["error", "issue", "problem", "bug", "fix", "broken", "not working"],
            "creative": ["design", "create", "imagine", "artistic", "creative", "draw"],
            "code": ["code", "program", "function", "class", "script", "algorithm"],
            "research": ["research", "find", "search", "investigate", "analyze"]
        }
        
        user_lower = user_input.lower()
        intent_scores = {}
        
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else "general"
        
        return {
            "primary_intent": primary_intent,
            "intent_scores": intent_scores,
            "confidence": max(intent_scores.values()) if intent_scores else 0.1
        }
    
    async def _extract_topic_patterns(
        self,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract topic patterns"""
        topic_keywords = {
            "programming": ["code", "program", "function", "class", "variable", "algorithm", "debug"],
            "web_development": ["website", "web", "html", "css", "javascript", "frontend", "backend"],
            "data_science": ["data", "analysis", "machine learning", "ai", "model", "dataset", "pandas"],
            "system_admin": ["server", "deploy", "infrastructure", "docker", "kubernetes", "linux"],
            "design": ["design", "ui", "ux", "interface", "layout", "visual", "graphics"],
            "business": ["business", "strategy", "market", "customer", "revenue", "growth"],
            "research": ["research", "study", "analysis", "investigation", "findings"],
            "education": ["learn", "tutorial", "course", "lesson", "teach", "explain"],
            "project": ["project", "build", "create", "develop", "implement"]
        }
        
        user_lower = user_input.lower()
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_lower)
            if score > 0:
                topic_scores[topic] = score / len(keywords)
        
        primary_topic = max(topic_scores.items(), key=lambda x: x[1])[0] if topic_scores else "general"
        
        return {
            "primary_topic": primary_topic,
            "topic_scores": topic_scores,
            "confidence": max(topic_scores.values()) if topic_scores else 0.1
        }
    
    async def _extract_complexity_patterns(
        self,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract complexity level patterns"""
        complexity_indicators = {
            "beginner": ["basic", "simple", "easy", "beginner", "start", "learn", "tutorial", "help"],
            "intermediate": ["intermediate", "moderate", "some experience", "familiar", "understand"],
            "advanced": ["advanced", "complex", "sophisticated", "expert", "professional", "optimize"],
            "expert": ["expert", "master", "architect", "senior", "lead", "enterprise", "scalable"]
        }
        
        user_lower = user_input.lower()
        complexity_scores = {}
        
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in user_lower)
            if score > 0:
                complexity_scores[level] = score / len(indicators)
        
        # Consider input length and technical terms
        input_length = len(user_input.split())
        technical_terms = len(re.findall(
            r'\b(api|database|framework|architecture|algorithm|optimization|scalability)\b',
            user_lower
        ))
        
        if input_length > 50 or technical_terms > 3:
            complexity_scores["advanced"] = complexity_scores.get("advanced", 0) + 0.3
        elif input_length < 10:
            complexity_scores["beginner"] = complexity_scores.get("beginner", 0) + 0.2
        
        primary_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0] if complexity_scores else "intermediate"
        
        return {
            "primary_complexity": primary_complexity,
            "complexity_scores": complexity_scores,
            "confidence": max(complexity_scores.values()) if complexity_scores else 0.1
        }
    
    async def _extract_style_patterns(
        self,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract communication style patterns"""
        style_indicators = {
            "formal": ["please", "could you", "would you", "thank you", "sir", "madam", "kindly"],
            "casual": ["hey", "hi", "yeah", "ok", "cool", "awesome", "thanks", "sup"],
            "technical": ["implement", "configure", "optimize", "debug", "refactor", "deploy"],
            "creative": ["design", "create", "imagine", "artistic", "beautiful", "elegant"],
            "urgent": ["urgent", "asap", "quickly", "immediately", "fast", "now", "hurry"],
            "detailed": ["detailed", "comprehensive", "thorough", "complete", "extensive"],
            "concise": ["brief", "short", "quick", "simple", "summary", "overview"]
        }
        
        user_lower = user_input.lower()
        style_scores = {}
        
        for style, indicators in style_indicators.items():
            score = sum(1 for indicator in indicators if indicator in user_lower)
            if score > 0:
                style_scores[style] = score / len(indicators)
        
        primary_style = max(style_scores.items(), key=lambda x: x[1])[0] if style_scores else "neutral"
        
        return {
            "primary_style": primary_style,
            "style_scores": style_scores,
            "confidence": max(style_scores.values()) if style_scores else 0.1
        }
    
    async def _extract_sentiment_patterns(
        self,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract sentiment patterns"""
        positive_words = ["good", "great", "excellent", "amazing", "perfect", "love", "like", "awesome", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "wrong", "error", "problem", "frustrated"]
        neutral_words = ["okay", "fine", "normal", "standard", "regular", "average"]
        
        user_lower = user_input.lower()
        
        positive_score = sum(1 for word in positive_words if word in user_lower)
        negative_score = sum(1 for word in negative_words if word in user_lower)
        neutral_score = sum(1 for word in neutral_words if word in user_lower)
        
        total_score = positive_score + negative_score + neutral_score
        
        if total_score == 0:
            sentiment = "neutral"
            confidence = 0.1
        else:
            if positive_score > negative_score and positive_score > neutral_score:
                sentiment = "positive"
                confidence = positive_score / total_score
            elif negative_score > positive_score and negative_score > neutral_score:
                sentiment = "negative"
                confidence = negative_score / total_score
            else:
                sentiment = "neutral"
                confidence = neutral_score / total_score if neutral_score > 0 else 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            }
        }
    
    async def _extract_context_patterns(
        self,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract context patterns"""
        if not context:
            return {"context_type": "none", "confidence": 0.1}
        
        context_patterns = {
            "project_context": bool(context.get("project_id")),
            "chat_history": bool(context.get("chat_id")),
            "task_specific": bool(context.get("task_type")),
            "user_specific": bool(context.get("user_id")),
            "time_sensitive": bool(context.get("deadline") or "urgent" in user_input.lower())
        }
        
        active_contexts = [k for k, v in context_patterns.items() if v]
        primary_context = active_contexts[0] if active_contexts else "general"
        
        return {
            "primary_context": primary_context,
            "active_contexts": active_contexts,
            "confidence": len(active_contexts) / len(context_patterns)
        }
    
    def _calculate_quality_score(
        self,
        ai_response: str,
        user_feedback: Optional[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> float:
        """Calculate quality score for a response"""
        base_score = 0.5
        
        # User feedback (most important)
        if user_feedback:
            rating = user_feedback.get("rating", 3)  # 1-5 scale
            feedback_score = rating / 5.0
            base_score += feedback_score * 0.4
            
            # Positive/negative feedback
            if user_feedback.get("helpful", False):
                base_score += 0.1
            if user_feedback.get("accurate", False):
                base_score += 0.1
        
        # Response characteristics
        response_length = len(ai_response.split())
        if 20 <= response_length <= 300:  # Optimal length
            base_score += 0.1
        elif response_length > 500:  # Too long
            base_score -= 0.05
        
        # Structure and formatting
        if any(marker in ai_response for marker in ["1.", "2.", "•", "-", "Step"]):
            base_score += 0.05
        
        # Code presence (if relevant)
        if "```" in ai_response and context and context.get("task_type") == "code_generation":
            base_score += 0.1
        
        # Helpful phrases
        helpful_phrases = ["here's how", "you can", "try this", "for example", "step by step"]
        helpful_count = sum(1 for phrase in helpful_phrases if phrase in ai_response.lower())
        base_score += min(helpful_count * 0.02, 0.1)
        
        return min(max(base_score, 0.0), 1.0)
    
    def _find_similar_patterns(
        self,
        patterns: Dict[str, Any],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Find similar patterns in knowledge base"""
        similar_patterns = []
        
        # Check user-specific patterns first
        user_patterns = self.knowledge_base["patterns"].get(user_id, {})
        for pattern_key, pattern_data in user_patterns.items():
            similarity = self._calculate_pattern_similarity(patterns, pattern_data["patterns"])
            if similarity > self.learning_config["pattern_similarity_threshold"]:
                similar_patterns.append({
                    "pattern_key": pattern_key,
                    "similarity": similarity,
                    "quality_score": pattern_data["quality_score"],
                    "usage_count": pattern_data.get("usage_count", 1),
                    "user_specific": True
                })
        
        # Check global patterns
        global_patterns = self.knowledge_base["patterns"].get("global", {})
        for pattern_key, pattern_data in global_patterns.items():
            similarity = self._calculate_pattern_similarity(patterns, pattern_data["patterns"])
            if similarity > self.learning_config["pattern_similarity_threshold"]:
                similar_patterns.append({
                    "pattern_key": pattern_key,
                    "similarity": similarity,
                    "quality_score": pattern_data["quality_score"],
                    "usage_count": pattern_data.get("usage_count", 1),
                    "user_specific": False
                })
        
        # Sort by similarity and quality
        similar_patterns.sort(
            key=lambda x: x["similarity"] * x["quality_score"],
            reverse=True
        )
        
        return similar_patterns[:10]  # Top 10 similar patterns
    
    def _calculate_pattern_similarity(
        self,
        patterns1: Dict[str, Any],
        patterns2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two pattern sets"""
        if not patterns1 or not patterns2:
            return 0.0
        
        similarities = []
        
        # Intent similarity
        if "intent" in patterns1 and "intent" in patterns2:
            intent1 = patterns1["intent"].get("primary_intent", "")
            intent2 = patterns2["intent"].get("primary_intent", "")
            similarities.append(1.0 if intent1 == intent2 else 0.0)
        
        # Topic similarity
        if "topic" in patterns1 and "topic" in patterns2:
            topic1 = patterns1["topic"].get("primary_topic", "")
            topic2 = patterns2["topic"].get("primary_topic", "")
            similarities.append(1.0 if topic1 == topic2 else 0.0)
        
        # Complexity similarity
        if "complexity" in patterns1 and "complexity" in patterns2:
            complexity1 = patterns1["complexity"].get("primary_complexity", "")
            complexity2 = patterns2["complexity"].get("primary_complexity", "")
            similarities.append(1.0 if complexity1 == complexity2 else 0.5)
        
        # Style similarity
        if "style" in patterns1 and "style" in patterns2:
            style1 = patterns1["style"].get("primary_style", "")
            style2 = patterns2["style"].get("primary_style", "")
            similarities.append(1.0 if style1 == style2 else 0.3)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_response_confidence(
        self,
        similar_patterns: List[Dict[str, Any]],
        current_patterns: Dict[str, Any]
    ) -> float:
        """Calculate confidence for generating a response"""
        if not similar_patterns:
            return 0.1
        
        # Base confidence from similarity
        avg_similarity = sum(p["similarity"] for p in similar_patterns) / len(similar_patterns)
        
        # Quality of similar patterns
        avg_quality = sum(p["quality_score"] for p in similar_patterns) / len(similar_patterns)
        
        # Usage frequency (more used patterns are more reliable)
        avg_usage = sum(p["usage_count"] for p in similar_patterns) / len(similar_patterns)
        usage_factor = min(avg_usage / 10, 1.0)  # Normalize to 0-1
        
        # User-specific patterns get bonus
        user_specific_count = sum(1 for p in similar_patterns if p["user_specific"])
        user_specific_factor = user_specific_count / len(similar_patterns)
        
        # Combine factors
        confidence = (
            avg_similarity * 0.3 +
            avg_quality * 0.3 +
            usage_factor * 0.2 +
            user_specific_factor * 0.2
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    async def _generate_learned_response(
        self,
        user_input: str,
        patterns: Dict[str, Any],
        similar_patterns: List[Dict[str, Any]],
        context: Dict[str, Any],
        user_id: str
    ) -> str:
        """Generate response using learned patterns"""
        try:
            # Get user preferences
            user_prefs = self.knowledge_base["user_preferences"].get(user_id, {})
            
            # Find best response template
            best_pattern = similar_patterns[0] if similar_patterns else None
            
            if not best_pattern:
                return "I'm still learning about this topic. Could you provide more details?"
            
            # Get response template
            template_key = best_pattern["pattern_key"]
            templates = self.knowledge_base["response_templates"].get(user_id, {}).get(template_key, [])
            
            if not templates:
                templates = self.knowledge_base["response_templates"].get("global", {}).get(template_key, [])
            
            if templates:
                # Use best template
                best_template = max(templates, key=lambda t: t.get("quality_score", 0))
                base_response = best_template["response"]
            else:
                base_response = "Based on similar requests, here's what I can help you with:"
            
            # Adapt response based on patterns and preferences
            adapted_response = await self._adapt_response(
                base_response, patterns, user_prefs, context
            )
            
            return adapted_response
            
        except Exception as e:
            logger.error(f"Learned response generation failed: {str(e)}")
            return "I'm having trouble generating a response based on my learning. Let me help you in a different way."
    
    async def _adapt_response(
        self,
        base_response: str,
        patterns: Dict[str, Any],
        user_prefs: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Adapt response based on patterns and preferences"""
        try:
            adapted = base_response
            
            # Adapt based on complexity level
            complexity = patterns.get("complexity", {}).get("primary_complexity", "intermediate")
            if complexity == "beginner":
                adapted = f"Let me explain this step by step:\n\n{adapted}\n\nFeel free to ask if you need clarification!"
            elif complexity == "advanced":
                adapted = f"Here's a comprehensive approach:\n\n{adapted}\n\nThis considers advanced optimization and best practices."
            
            # Adapt based on style preferences
            preferred_style = user_prefs.get("communication_style", patterns.get("style", {}).get("primary_style", "neutral"))
            
            if preferred_style == "formal":
                adapted = adapted.replace("you can", "you may")
                adapted = adapted.replace("here's", "here is")
            elif preferred_style == "casual":
                if not adapted.startswith(("Hey", "Hi")):
                    adapted = f"Hey! {adapted}"
            
            # Adapt based on sentiment
            sentiment = patterns.get("sentiment", {}).get("sentiment", "neutral")
            if sentiment == "negative":
                adapted = f"I understand this might be frustrating. {adapted}\n\nI'm here to help make this easier for you."
            elif sentiment == "positive":
                adapted = f"Great question! {adapted}\n\nI'm glad I could help with this!"
            
            # Add context-specific information
            if context.get("project_id"):
                adapted += f"\n\n(Note: This is for your project {context['project_id']})"
            
            return adapted
            
        except Exception as e:
            logger.error(f"Response adaptation failed: {str(e)}")
            return base_response
    
    async def _update_knowledge_immediately(
        self,
        user_id: str,
        patterns: Dict[str, Any],
        response: str,
        quality_score: float
    ):
        """Update knowledge base immediately for high-quality interactions"""
        try:
            pattern_key = self._generate_pattern_key(patterns)
            
            # Update user-specific patterns
            if user_id not in self.knowledge_base["patterns"]:
                self.knowledge_base["patterns"][user_id] = {}
            
            if pattern_key not in self.knowledge_base["patterns"][user_id]:
                self.knowledge_base["patterns"][user_id][pattern_key] = {
                    "patterns": patterns,
                    "quality_score": quality_score,
                    "usage_count": 1,
                    "last_used": datetime.now().isoformat()
                }
            else:
                # Update existing pattern
                existing = self.knowledge_base["patterns"][user_id][pattern_key]
                existing["usage_count"] += 1
                existing["quality_score"] = (
                    (existing["quality_score"] * (existing["usage_count"] - 1) + quality_score) /
                    existing["usage_count"]
                )
                existing["last_used"] = datetime.now().isoformat()
            
            # Update response templates
            if user_id not in self.knowledge_base["response_templates"]:
                self.knowledge_base["response_templates"][user_id] = {}
            
            if pattern_key not in self.knowledge_base["response_templates"][user_id]:
                self.knowledge_base["response_templates"][user_id][pattern_key] = []
            
            self.knowledge_base["response_templates"][user_id][pattern_key].append({
                "response": response,
                "quality_score": quality_score,
                "created_at": datetime.now().isoformat()
            })
            
            # Limit template history
            templates = self.knowledge_base["response_templates"][user_id][pattern_key]
            if len(templates) > 10:
                # Keep only the best templates
                templates.sort(key=lambda t: t["quality_score"], reverse=True)
                self.knowledge_base["response_templates"][user_id][pattern_key] = templates[:10]
            
        except Exception as e:
            logger.error(f"Immediate knowledge update failed: {str(e)}")
    
    async def _update_pattern_knowledge(self, patterns: Dict[str, Any], quality_score: float):
        """Update pattern knowledge in the knowledge base"""
        try:
            pattern_key = self._generate_pattern_key(patterns)
            
            # Update global patterns
            if "global" not in self.knowledge_base["patterns"]:
                self.knowledge_base["patterns"]["global"] = {}
            
            if pattern_key not in self.knowledge_base["patterns"]["global"]:
                self.knowledge_base["patterns"]["global"][pattern_key] = {
                    "patterns": patterns,
                    "quality_score": quality_score,
                    "usage_count": 1,
                    "first_seen": datetime.now().isoformat()
                }
                self.learning_stats["patterns_discovered"] += 1
            else:
                # Update existing pattern
                existing = self.knowledge_base["patterns"]["global"][pattern_key]
                old_quality = existing["quality_score"]
                existing["usage_count"] += 1
                existing["quality_score"] = (
                    (existing["quality_score"] * (existing["usage_count"] - 1) + quality_score) /
                    existing["usage_count"]
                )
                
                # Track quality improvements
                if existing["quality_score"] > old_quality + self.learning_config["quality_improvement_threshold"]:
                    self.learning_stats["quality_improvements"] += 1
            
        except Exception as e:
            logger.error(f"Pattern knowledge update failed: {str(e)}")
    
    async def _update_user_preferences(
        self,
        user_id: str,
        patterns: Dict[str, Any],
        quality_score: float
    ):
        """Update user-specific preferences"""
        try:
            if user_id not in self.knowledge_base["user_preferences"]:
                self.knowledge_base["user_preferences"][user_id] = {
                    "communication_style": "neutral",
                    "complexity_preference": "intermediate",
                    "topic_interests": Counter(),
                    "response_length_preference": "medium",
                    "interaction_count": 0,
                    "average_satisfaction": 0.5
                }
            
            user_prefs = self.knowledge_base["user_preferences"][user_id]
            user_prefs["interaction_count"] += 1
            
            # Update average satisfaction
            old_avg = user_prefs["average_satisfaction"]
            count = user_prefs["interaction_count"]
            user_prefs["average_satisfaction"] = (old_avg * (count - 1) + quality_score) / count
            
            # Update style preference (if high quality)
            if quality_score > 0.7:
                style = patterns.get("style", {}).get("primary_style", "neutral")
                if style != "neutral":
                    user_prefs["communication_style"] = style
                
                complexity = patterns.get("complexity", {}).get("primary_complexity", "intermediate")
                user_prefs["complexity_preference"] = complexity
            
            # Update topic interests
            topic = patterns.get("topic", {}).get("primary_topic", "general")
            user_prefs["topic_interests"][topic] += 1
            
        except Exception as e:
            logger.error(f"User preferences update failed: {str(e)}")
    
    async def _update_response_templates(
        self,
        user_input: str,
        ai_response: str,
        patterns: Dict[str, Any],
        quality_score: float
    ):
        """Update response templates"""
        try:
            if quality_score < 0.6:  # Only store good responses
                return
            
            pattern_key = self._generate_pattern_key(patterns)
            
            # Update global templates
            if "global" not in self.knowledge_base["response_templates"]:
                self.knowledge_base["response_templates"]["global"] = {}
            
            if pattern_key not in self.knowledge_base["response_templates"]["global"]:
                self.knowledge_base["response_templates"]["global"][pattern_key] = []
            
            self.knowledge_base["response_templates"]["global"][pattern_key].append({
                "input": user_input,
                "response": ai_response,
                "quality_score": quality_score,
                "patterns": patterns,
                "created_at": datetime.now().isoformat()
            })
            
            # Limit template history
            templates = self.knowledge_base["response_templates"]["global"][pattern_key]
            if len(templates) > 20:
                # Keep only the best templates
                templates.sort(key=lambda t: t["quality_score"], reverse=True)
                self.knowledge_base["response_templates"]["global"][pattern_key] = templates[:20]
            
        except Exception as e:
            logger.error(f"Response templates update failed: {str(e)}")
    
    async def _generate_improvement_suggestions(
        self,
        user_input: str,
        ai_response: str,
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions for low-quality responses"""
        try:
            improvement_prompt = f"""
            Analyze this AI response and suggest improvements:
            
            User Input: "{user_input}"
            AI Response: "{ai_response}"
            Detected Patterns: {json.dumps(patterns, indent=2)}
            
            Suggest 3-5 specific improvements:
            1. Content improvements
            2. Structure improvements
            3. Style improvements
            4. Completeness improvements
            5. Accuracy improvements
            
            Return as JSON array of improvement suggestions.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=improvement_prompt,
                task_type="analytical",
                context={"improvement_analysis": True}
            )
            
            try:
                suggestions = json.loads(response.get("text", "[]"))
                return suggestions if isinstance(suggestions, list) else [suggestions]
            except json.JSONDecodeError:
                return [response.get("text", "")]
                
        except Exception as e:
            logger.error(f"Improvement suggestions generation failed: {str(e)}")
            return []
    
    def _generate_pattern_key(self, patterns: Dict[str, Any]) -> str:
        """Generate a unique key for pattern set"""
        key_components = []
        
        for pattern_type in ["intent", "topic", "complexity", "style"]:
            if pattern_type in patterns:
                primary_key = f"primary_{pattern_type}"
                if primary_key in patterns[pattern_type]:
                    key_components.append(f"{pattern_type}:{patterns[pattern_type][primary_key]}")
        
        key_string = "|".join(key_components) if key_components else "general"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    async def _load_knowledge_base(self):
        """Load knowledge base from database"""
        try:
            # Load recent high-quality interactions
            interactions = await self.db_manager.get_unprocessed_learning_interactions(limit=1000)
            
            processed_count = 0
            for interaction in interactions:
                if interaction.quality_score > 0.7:  # Only load high-quality interactions
                    await self._process_single_interaction(interaction)
                    processed_count += 1
            
            self.learning_stats["knowledge_base_size"] = len(self.knowledge_base["patterns"])
            
            logger.info(f"Loaded knowledge base with {processed_count} high-quality interactions")
            
        except Exception as e:
            logger.error(f"Knowledge base loading failed: {str(e)}")
    
    async def _save_knowledge_base(self):
        """Save knowledge base to persistent storage"""
        try:
            # In a production system, this would save to a persistent store
            # For now, we rely on database storage of interactions
            
            self.learning_stats["knowledge_base_size"] = len(self.knowledge_base["patterns"])
            logger.info(f"Knowledge base saved with {self.learning_stats['knowledge_base_size']} pattern groups")
            
        except Exception as e:
            logger.error(f"Knowledge base saving failed: {str(e)}")
    
    async def _optimize_knowledge_base(self):
        """Optimize knowledge base by removing low-quality patterns"""
        try:
            # Remove patterns with low quality and usage
            for user_id in list(self.knowledge_base["patterns"].keys()):
                patterns_to_remove = []
                
                for pattern_key, pattern_data in self.knowledge_base["patterns"][user_id].items():
                    if (pattern_data["quality_score"] < 0.3 and 
                        pattern_data.get("usage_count", 0) < 2):
                        patterns_to_remove.append(pattern_key)
                
                for pattern_key in patterns_to_remove:
                    del self.knowledge_base["patterns"][user_id][pattern_key]
                
                # Remove empty user entries
                if not self.knowledge_base["patterns"][user_id]:
                    del self.knowledge_base["patterns"][user_id]
            
            # Limit response templates
            for user_id in self.knowledge_base["response_templates"]:
                for pattern_key in self.knowledge_base["response_templates"][user_id]:
                    templates = self.knowledge_base["response_templates"][user_id][pattern_key]
                    if len(templates) > 15:
                        # Keep only the best templates
                        templates.sort(key=lambda t: t["quality_score"], reverse=True)
                        self.knowledge_base["response_templates"][user_id][pattern_key] = templates[:15]
            
            logger.info("Knowledge base optimization completed")
            
        except Exception as e:
            logger.error(f"Knowledge base optimization failed: {str(e)}")
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "learning_stats": self.learning_stats,
            "knowledge_base": {
                "total_patterns": sum(len(patterns) for patterns in self.knowledge_base["patterns"].values()),
                "user_specific_patterns": len(self.knowledge_base["patterns"]) - 1,  # Exclude global
                "response_templates": sum(
                    len(templates) for user_templates in self.knowledge_base["response_templates"].values()
                    for templates in user_templates.values()
                ),
                "user_preferences": len(self.knowledge_base["user_preferences"])
            },
            "service_status": {
                "is_running": self.is_running,
                "learning_enabled": self.settings.ENABLE_SELF_LEARNING,
                "last_optimization": self.learning_stats.get("last_optimization"),
                "background_task_active": self.learning_task is not None and not self.learning_task.done()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for self-learning service"""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "learning_active": self.is_running and self.learning_task and not self.learning_task.done(),
            "knowledge_base_loaded": len(self.knowledge_base["patterns"]) > 0,
            "total_patterns": sum(len(patterns) for patterns in self.knowledge_base["patterns"].values()),
            "learning_stats": self.learning_stats
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.is_running = False
            
            if self.learning_task:
                self.learning_task.cancel()
                try:
                    await self.learning_task
                except asyncio.CancelledError:
                    pass
            
            # Save final state
            await self._save_knowledge_base()
            
            logger.info("Self-Learning Service cleanup completed")
            
        except Exception as e:
            logger.error(f"Self-Learning Service cleanup failed: {str(e)}")
