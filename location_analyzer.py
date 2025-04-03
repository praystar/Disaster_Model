import spacy
from collections import Counter
from typing import List, Tuple, Dict
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

class LocationAnalyzer:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        # Initialize geocoder
        self.geocoder = Nominatim(user_agent="disaster_analyzer")
        # Cache for geocoding results
        self.location_cache: Dict[str, Dict] = {}
        
    def _get_location_info(self, location: str) -> Dict:
        """
        Get location information using geocoding
        """
        if location in self.location_cache:
            return self.location_cache[location]
            
        try:
            # Add delay to respect geocoding service rate limits
            time.sleep(1)
            location_info = self.geocoder.geocode(location, language="en")
            if location_info:
                result = {
                    'latitude': location_info.latitude,
                    'longitude': location_info.longitude,
                    'importance': location_info.raw.get('importance', 0),
                    'type': location_info.raw.get('type', ''),
                    'class': location_info.raw.get('class', ''),
                    'display_name': location_info.address
                }
            else:
                result = None
        except (GeocoderTimedOut, Exception):
            result = None
            
        self.location_cache[location] = result
        return result

    def _analyze_location_context(self, text: str, location: str) -> float:
        """
        Analyze the context around a location mention to determine its relevance
        """
        doc = self.nlp(text)
        location_lower = location.lower()
        
        # Context keywords that indicate primary disaster location
        location_indicators = [
            "struck", "hit", "occurred", "happened in", "affected",
            "devastated", "impacted", "at", "in", "near"
        ]
        
        # Find all mentions of the location and analyze their context
        relevance_score = 0
        for token in doc:
            if token.text.lower() in location_lower:
                # Check surrounding words (window of 5 tokens)
                context_window = doc[max(0, token.i - 5):min(len(doc), token.i + 6)]
                
                # Check for location indicators in context
                for word in context_window:
                    if word.text.lower() in location_indicators:
                        relevance_score += 1
                        
                # Check if location is in the first sentence (likely more important)
                if token in next(doc.sents):
                    relevance_score += 2
                    
        return relevance_score

    def determine_primary_location(self, text: str, locations: List[str]) -> str:
        """
        Determine the primary disaster location from a list of candidates
        """
        if not locations:
            return None
            
        if len(locations) == 1:
            return locations[0]
            
        location_scores: Dict[str, float] = {}
        
        for location in locations:
            score = 0
            
            # 1. Get location information
            location_info = self._get_location_info(location)
            
            if location_info:
                # Add score based on location importance from geocoding
                score += float(location_info.get('importance', 0)) * 10
                
                # Add score based on location type
                if location_info.get('type') in ['city', 'town', 'village']:
                    score += 5  # More specific locations get higher scores
                elif location_info.get('type') in ['state', 'province']:
                    score += 3
                elif location_info.get('type') in ['country']:
                    score += 1
            
            # 2. Analyze mention frequency
            location_lower = location.lower()
            mention_count = sum(1 for loc in text.lower().split() if loc in location_lower)
            score += mention_count * 2
            
            # 3. Analyze context
            context_score = self._analyze_location_context(text, location)
            score += context_score * 3
            
            location_scores[location] = score
        
        # Get the location with the highest score
        primary_location = max(location_scores.items(), key=lambda x: x[1])[0]
        
        return primary_location 