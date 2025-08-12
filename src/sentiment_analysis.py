"""
Sentiment Analysis Module for VancouverPy

This module provides multilingual sentiment analysis capabilities for restaurant reviews
using state-of-the-art transformer models.

Author: VancouverPy Project Team
Date: August 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Sentiment analysis will use fallback method.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilinguaSentimentAnalyzer:
    """Multilingual sentiment analysis using transformers"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Fallback to English-only
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Transformers not available. Using simple fallback sentiment analysis.")
    
    def _load_model(self):
        """Load the sentiment analysis model"""
        try:
            # Try multilingual model first
            multilingual_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            logger.info(f"Loading multilingual sentiment model: {multilingual_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(multilingual_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(multilingual_model)
            self.model.to(self.device)
            self.model_name = multilingual_model
            
            logger.info(f"Successfully loaded multilingual model on {self.device}")
            
        except Exception as e:
            logger.warning(f"Failed to load multilingual model: {e}")
            try:
                # Fallback to English-only model
                logger.info(f"Loading English sentiment model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                
                logger.info(f"Successfully loaded English model on {self.device}")
                
            except Exception as e2:
                logger.error(f"Failed to load any sentiment model: {e2}")
                self.model = None
                self.tokenizer = None
    
    def predict_sentiment(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Predict sentiment for a list of texts
        
        Args:
            texts: List of text strings to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of dictionaries with sentiment predictions
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return self._fallback_sentiment(texts)
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._process_batch(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, texts: List[str]) -> List[Dict]:
        """Process a batch of texts"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert to probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Map predictions to sentiment labels
            results = []
            for i, (text, probs) in enumerate(zip(texts, probabilities)):
                predicted_class = torch.argmax(probs).item()
                confidence = torch.max(probs).item()
                
                # Map class to sentiment (model-dependent)
                if "roberta" in self.model_name.lower():
                    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                else:
                    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                
                sentiment = sentiment_map.get(predicted_class, "Neutral")
                
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'sentiment_score': self._convert_to_score(sentiment, confidence)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [{'text': text, 'sentiment': 'Neutral', 'confidence': 0.5, 'sentiment_score': 0.5} 
                   for text in texts]
    
    def _convert_to_score(self, sentiment: str, confidence: float) -> float:
        """Convert sentiment label to numerical score (0-1)"""
        base_scores = {
            'Negative': 0.2,
            'Neutral': 0.5,
            'Positive': 0.8
        }
        
        base_score = base_scores.get(sentiment, 0.5)
        
        # Adjust based on confidence
        if sentiment == 'Positive':
            return min(1.0, base_score + (confidence - 0.5) * 0.4)
        elif sentiment == 'Negative':
            return max(0.0, base_score - (confidence - 0.5) * 0.4)
        else:
            return base_score
    
    def _fallback_sentiment(self, texts: List[str]) -> List[Dict]:
        """Simple fallback sentiment analysis using keyword matching"""
        logger.info("Using fallback sentiment analysis")
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 
                         'fantastic', 'wonderful', 'perfect', 'delicious', 'outstanding']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting',
                         'disappointing', 'poor', 'rude', 'slow', 'expensive']
        
        results = []
        for text in texts:
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = 'Positive'
                score = 0.7
            elif negative_count > positive_count:
                sentiment = 'Negative'
                score = 0.3
            else:
                sentiment = 'Neutral'
                score = 0.5
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': 0.6,
                'sentiment_score': score
            })
        
        return results
    
    def analyze_restaurant_reviews(self, reviews_df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze sentiment for restaurant reviews in a DataFrame
        
        Args:
            reviews_df: DataFrame containing reviews
            text_column: Name of the column containing review text
            
        Returns:
            DataFrame with added sentiment columns
        """
        if reviews_df.empty or text_column not in reviews_df.columns:
            logger.warning(f"No valid reviews found in column '{text_column}'")
            return reviews_df
        
        logger.info(f"Analyzing sentiment for {len(reviews_df)} reviews...")
        
        # Clean and prepare texts
        texts = reviews_df[text_column].fillna('').astype(str).tolist()
        
        # Analyze sentiment
        sentiment_results = self.predict_sentiment(texts)
        
        # Add results to DataFrame
        reviews_df = reviews_df.copy()
        reviews_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
        reviews_df['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]
        reviews_df['sentiment_score'] = [r['sentiment_score'] for r in sentiment_results]
        
        # Calculate summary statistics
        sentiment_summary = reviews_df['sentiment'].value_counts()
        avg_sentiment_score = reviews_df['sentiment_score'].mean()
        
        logger.info(f"Sentiment analysis complete:")
        logger.info(f"  Average sentiment score: {avg_sentiment_score:.3f}")
        logger.info(f"  Sentiment distribution: {dict(sentiment_summary)}")
        
        return reviews_df

def demo_multilingual_sentiment():
    """Demonstrate multilingual sentiment analysis"""
    analyzer = MultilinguaSentimentAnalyzer()
    
    texts = [
        # English
        "I absolutely love the new design of this app!",
        "The customer service was disappointing.",
        "The weather is fine, nothing special.",
        
        # Chinese
        "这家餐厅的菜味道非常棒！",
        "我对他的回答很失望。",
        "天气今天一般。",
        
        # Spanish
        "¡Me encanta cómo quedó la decoración!",
        "El servicio fue terrible y muy lento.",
        "El libro estuvo más o menos.",
        
        # French
        "J'adore ce restaurant, c'est excellent !",
        "L'attente était trop longue et frustrante.",
        "Le film était moyen, sans plus.",
        
        # Japanese
        "このレストランの料理は本当に美味しいです！",
        "このホテルのサービスはがっかりしました。",
        "天気はまあまあです。"
    ]
    
    print("🌍 Multilingual Sentiment Analysis Demo")
    print("=" * 50)
    
    results = analyzer.predict_sentiment(texts)
    
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"Score: {result['sentiment_score']:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    demo_multilingual_sentiment()
