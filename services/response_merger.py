"""
Response merger for combining multiple LLM responses intelligently
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import difflib

from core.logger import get_logger

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

class ResponseMerger:
    """Intelligently merge responses from multiple LLMs"""
    
    def __init__(self, settings):
        self.settings = settings
        
        # Merge strategies
        self.strategies = {
            'best_quality': self._merge_best_quality,
            'consensus': self._merge_consensus,
            'weighted_average': self._merge_weighted_average,
            'code_specific': self._merge_code_specific,
            'creative_blend': self._merge_creative_blend
        }
        
        # Task-specific merge preferences
        self.task_strategies = {
            'code_generation': 'code_specific',
            'code_fix': 'consensus',
            'code_review': 'weighted_average',
            'creative': 'creative_blend',
            'analytical': 'consensus',
            'conversational': 'best_quality'
        }
    
    async def merge_responses(
        self,
        responses: List[LLMResponse],
        task_type: str = "conversational",
        require_consensus: bool = False
    ) -> Dict[str, Any]:
        """Merge multiple LLM responses intelligently"""
        
        if not responses:
            raise ValueError("No responses to merge")
        
        if len(responses) == 1:
            return {
                'text': responses[0].text,
                'confidence': responses[0].quality_score,
                'sources': [responses[0].model],
                'strategy': 'single_response',
                'consensus': True
            }
        
        # Select merge strategy
        strategy = self.task_strategies.get(task_type, 'best_quality')
        
        if require_consensus:
            strategy = 'consensus'
        
        # Apply merge strategy
        merge_func = self.strategies[strategy]
        result = await merge_func(responses, task_type)
        
        # Add metadata
        result['strategy'] = strategy
        result['sources'] = [r.model for r in responses]
        result['response_count'] = len(responses)
        
        return result
    
    async def _merge_best_quality(
        self,
        responses: List[LLMResponse],
        task_type: str
    ) -> Dict[str, Any]:
        """Select the highest quality response"""
        
        best_response = max(responses, key=lambda r: r.quality_score)
        
        return {
            'text': best_response.text,
            'confidence': best_response.quality_score,
            'consensus': False,
            'primary_model': best_response.model
        }
    
    async def _merge_consensus(
        self,
        responses: List[LLMResponse],
        task_type: str
    ) -> Dict[str, Any]:
        """Find consensus among responses"""
        
        # Calculate similarity between responses
        similarities = []
        for i, resp1 in enumerate(responses):
            for j, resp2 in enumerate(responses[i+1:], i+1):
                similarity = self._calculate_similarity(resp1.text, resp2.text)
                similarities.append((i, j, similarity))
        
        # Find the most similar pair
        if similarities:
            similarities.sort(key=lambda x: x[2], reverse=True)
            best_pair = similarities[0]
            
            if best_pair[2] > 0.7:  # High similarity threshold
                # Merge the most similar responses
                resp1 = responses[best_pair[0]]
                resp2 = responses[best_pair[1]]
                
                merged_text = await self._merge_similar_texts(resp1.text, resp2.text)
                confidence = (resp1.quality_score + resp2.quality_score) / 2
                
                return {
                    'text': merged_text,
                    'confidence': confidence,
                    'consensus': True,
                    'similarity_score': best_pair[2]
                }
        
        # No consensus found, return best quality
        return await self._merge_best_quality(responses, task_type)
    
    async def _merge_weighted_average(
        self,
        responses: List[LLMResponse],
        task_type: str
    ) -> Dict[str, Any]:
        """Merge responses using weighted averaging"""
        
        # Weight responses by quality score
        total_weight = sum(r.quality_score for r in responses)
        
        if total_weight == 0:
            return await self._merge_best_quality(responses, task_type)
        
        # For text, we'll use the highest quality response as base
        # and incorporate insights from others
        base_response = max(responses, key=lambda r: r.quality_score)
        
        # Extract key insights from other responses
        insights = []
        for response in responses:
            if response != base_response:
                insight = await self._extract_unique_insights(
                    base_response.text, response.text
                )
                if insight:
                    insights.append(insight)
        
        # Combine base response with insights
        merged_text = base_response.text
        if insights:
            merged_text += "\n\nAdditional insights:\n" + "\n".join(insights)
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            r.quality_score * (r.quality_score / total_weight) 
            for r in responses
        )
        
        return {
            'text': merged_text,
            'confidence': weighted_confidence,
            'consensus': False,
            'insights_added': len(insights)
        }
    
    async def _merge_code_specific(
        self,
        responses: List[LLMResponse],
        task_type: str
    ) -> Dict[str, Any]:
        """Merge code responses with special handling"""
        
        # Extract code blocks from each response
        code_blocks = []
        explanations = []
        
        for response in responses:
            code = self._extract_code_blocks(response.text)
            explanation = self._extract_explanation(response.text)
            
            if code:
                code_blocks.extend(code)
            if explanation:
                explanations.append(explanation)
        
        if not code_blocks:
            return await self._merge_best_quality(responses, task_type)
        
        # Select best code block (longest and most complete)
        best_code = max(code_blocks, key=lambda c: len(c['code']))
        
        # Combine explanations
        combined_explanation = await self._merge_explanations(explanations)
        
        # Format final response
        merged_text = f"```{best_code.get('language', '')}\n{best_code['code']}\n```"
        if combined_explanation:
            merged_text = combined_explanation + "\n\n" + merged_text
        
        return {
            'text': merged_text,
            'confidence': 0.8,  # High confidence for code
            'consensus': len(code_blocks) > 1,
            'code_language': best_code.get('language', 'unknown'),
            'code_blocks_found': len(code_blocks)
        }
    
    async def _merge_creative_blend(
        self,
        responses: List[LLMResponse],
        task_type: str
    ) -> Dict[str, Any]:
        """Creatively blend responses for creative tasks"""
        
        # For creative tasks, we want to combine the best elements
        # from each response
        
        # Split responses into sentences
        all_sentences = []
        for response in responses:
            sentences = self._split_into_sentences(response.text)
            for sentence in sentences:
                all_sentences.append({
                    'text': sentence,
                    'quality': response.quality_score,
                    'model': response.model
                })
        
        # Sort sentences by quality and uniqueness
        unique_sentences = []
        seen_content = set()
        
        for sentence_data in sorted(all_sentences, key=lambda x: x['quality'], reverse=True):
            sentence = sentence_data['text'].strip()
            if sentence and sentence not in seen_content:
                unique_sentences.append(sentence_data)
                seen_content.add(sentence)
        
        # Take the best sentences up to a reasonable length
        selected_sentences = unique_sentences[:10]  # Limit to 10 sentences
        
        # Combine into coherent text
        merged_text = ' '.join([s['text'] for s in selected_sentences])
        
        # Calculate average confidence
        avg_confidence = sum(s['quality'] for s in selected_sentences) / len(selected_sentences)
        
        return {
            'text': merged_text,
            'confidence': avg_confidence,
            'consensus': False,
            'sentences_combined': len(selected_sentences),
            'models_used': list(set(s['model'] for s in selected_sentences))
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()
    
    async def _merge_similar_texts(self, text1: str, text2: str) -> str:
        """Merge two similar texts"""
        
        # Split into sentences
        sentences1 = self._split_into_sentences(text1)
        sentences2 = self._split_into_sentences(text2)
        
        # Find common and unique sentences
        common_sentences = []
        unique_sentences = []
        
        for sentence in sentences1:
            if any(self._calculate_similarity(sentence, s2) > 0.8 for s2 in sentences2):
                if sentence not in common_sentences:
                    common_sentences.append(sentence)
            else:
                unique_sentences.append(sentence)
        
        # Add unique sentences from text2
        for sentence in sentences2:
            if not any(self._calculate_similarity(sentence, s1) > 0.8 for s1 in sentences1):
                unique_sentences.append(sentence)
        
        # Combine all sentences
        all_sentences = common_sentences + unique_sentences
        return ' '.join(all_sentences)
    
    async def _extract_unique_insights(self, base_text: str, comparison_text: str) -> Optional[str]:
        """Extract unique insights from comparison text"""
        
        base_sentences = set(self._split_into_sentences(base_text))
        comparison_sentences = self._split_into_sentences(comparison_text)
        
        unique_insights = []
        for sentence in comparison_sentences:
            if not any(self._calculate_similarity(sentence, base_s) > 0.7 for base_s in base_sentences):
                if len(sentence.strip()) > 20:  # Only meaningful sentences
                    unique_insights.append(sentence.strip())
        
        return ' '.join(unique_insights) if unique_insights else None
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from text"""
        
        code_blocks = []
        
        # Pattern for fenced code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            language = match[0] if match[0] else 'unknown'
            code = match[1].strip()
            
            if code:
                code_blocks.append({
                    'language': language,
                    'code': code
                })
        
        # Also look for inline code
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_pattern, text)
        
        for inline_code in inline_matches:
            if len(inline_code) > 10:  # Only longer code snippets
                code_blocks.append({
                    'language': 'unknown',
                    'code': inline_code
                })
        
        return code_blocks
    
    def _extract_explanation(self, text: str) -> Optional[str]:
        """Extract explanation text (non-code parts)"""
        
        # Remove code blocks
        text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text_without_code = re.sub(r'`[^`]+`', '', text_without_code)
        
        # Clean up
        explanation = text_without_code.strip()
        
        return explanation if len(explanation) > 20 else None
    
    async def _merge_explanations(self, explanations: List[str]) -> str:
        """Merge multiple explanations"""
        
        if not explanations:
            return ""
        
        if len(explanations) == 1:
            return explanations[0]
        
        # Find the most comprehensive explanation as base
        base_explanation = max(explanations, key=len)
        
        # Add unique points from other explanations
        unique_points = []
        base_sentences = set(self._split_into_sentences(base_explanation))
        
        for explanation in explanations:
            if explanation != base_explanation:
                sentences = self._split_into_sentences(explanation)
                for sentence in sentences:
                    if not any(self._calculate_similarity(sentence, base_s) > 0.7 for base_s in base_sentences):
                        unique_points.append(sentence)
        
        # Combine
        result = base_explanation
        if unique_points:
            result += "\n\nAdditional notes:\n" + "\n".join(f"• {point}" for point in unique_points[:3])
        
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
