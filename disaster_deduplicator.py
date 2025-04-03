from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from location_analyzer import LocationAnalyzer

class DisasterDeduplicator:
    def __init__(self, similarity_threshold=0.6, time_window_days=3):
        self.similarity_threshold = similarity_threshold
        self.time_window_days = time_window_days
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.location_analyzer = LocationAnalyzer()
        
    def _calculate_text_similarity(self, texts):
        """
        Calculate cosine similarity between all pairs of texts
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix)
    
    def _are_locations_same(self, loc1, loc2):
        """
        Check if two locations are effectively the same
        """
        if not loc1 or not loc2:
            return False
            
        # Clean and normalize locations
        loc1 = loc1.lower().strip()
        loc2 = loc2.lower().strip()
        
        # Direct match
        if loc1 == loc2:
            return True
            
        # Get location info from cache
        loc1_info = self.location_analyzer._get_location_info(loc1)
        loc2_info = self.location_analyzer._get_location_info(loc2)
        
        if loc1_info and loc2_info:
            # Check if coordinates are close (within ~50km)
            lat1, lon1 = loc1_info['latitude'], loc1_info['longitude']
            lat2, lon2 = loc2_info['latitude'], loc2_info['longitude']
            
            # Rough distance calculation (if within ~50km consider same location)
            if abs(lat1 - lat2) < 0.5 and abs(lon1 - lon2) < 0.5:
                return True
                
            # Check if one location contains the other in its display name
            if (loc1 in loc2_info['display_name'].lower() or 
                loc2 in loc1_info['display_name'].lower()):
                return True
                
        return False

    def _merge_disaster_data(self, disasters):
        """
        Merge multiple disaster entries into one
        """
        if len(disasters) == 1:
            return disasters[0]
            
        merged = disasters[0].copy()
        
        # Sum article counts
        merged['article_count'] = sum(d['article_count'] for d in disasters)
        
        # Combine URLs
        all_urls = []
        for d in disasters:
            if isinstance(d['urls'], list):
                all_urls.extend(d['urls'])
            else:
                all_urls.extend(eval(d['urls']))
        merged['urls'] = list(set(all_urls))  # Remove duplicates
        
        # Get the most recent date
        all_dates = [self._parse_date(d['published_at']) for d in disasters]
        merged['published_at'] = max(all_dates)
        
        # Combine texts
        all_texts = [d['text'] for d in disasters]
        merged['text'] = ' '.join(all_texts)
        
        # Take the highest severity
        severity_rank = {'high': 3, 'medium': 2, 'low': 1, 'unknown': 0}
        merged['severity'] = max(
            [d['severity'] for d in disasters],
            key=lambda x: severity_rank.get(x, 0)
        )
        
        # For disaster type, take the most specific one or most common
        disaster_types = [d['disaster_type'] for d in disasters if d['disaster_type'] != 'unknown']
        if disaster_types:
            merged['disaster_type'] = max(set(disaster_types), key=disaster_types.count)
        
        return merged

    def _parse_date(self, date_str):
        """
        Parse date string to datetime object
        """
        if isinstance(date_str, str):
            try:
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return date_str
    
    def _combine_locations(self, locations_lists, combined_text):
        """
        Determine primary location from all mentioned locations
        """
        # Gather all unique locations
        all_locations = set()
        for locs in locations_lists:
            if isinstance(locs, str):
                locs = eval(locs)
            if isinstance(locs, list):
                all_locations.update(locs)
        
        # Determine primary location
        primary_location = self.location_analyzer.determine_primary_location(
            combined_text,
            list(all_locations)
        )
        
        return primary_location
    
    def combine_duplicate_disasters(self, df):
        """
        Combine duplicate disaster reports based on text similarity,
        location overlap, and temporal proximity
        """
        # First pass: combine based on text similarity and temporal proximity
        texts = df['text'].tolist()
        similarity_matrix = self._calculate_text_similarity(texts)
        
        # Create initial clusters
        clusters = defaultdict(list)
        processed = set()
        
        for i in range(len(df)):
            if i in processed:
                continue
                
            cluster = [i]
            processed.add(i)
            
            for j in range(i + 1, len(df)):
                if j in processed:
                    continue
                    
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    date_i = self._parse_date(df.iloc[i]['published_at'])
                    date_j = self._parse_date(df.iloc[j]['published_at'])
                    days_diff = abs((date_j - date_i).days)
                    
                    if days_diff <= self.time_window_days:
                        cluster.append(j)
                        processed.add(j)
            
            cluster_id = min(cluster)
            clusters[cluster_id].extend(cluster)
        
        # Process initial clusters
        initial_combined = []
        
        for cluster_indices in clusters.values():
            cluster_data = df.iloc[cluster_indices]
            
            # Combine all texts first
            combined_text = ' '.join(cluster_data['text'])
            
            # Determine primary location
            primary_location = self._combine_locations(
                cluster_data['locations'],
                combined_text
            )
            
            # Create combined disaster entry
            combined_disaster = {
                'disaster_type': self._get_most_common_value(cluster_data['disaster_type']),
                'severity': self._get_highest_severity(cluster_data['severity']),
                'location': primary_location,
                'published_at': max(cluster_data['published_at'].apply(self._parse_date)),
                'urls': list(cluster_data['url']),
                'text': combined_text,
                'article_count': len(cluster_indices)
            }
            
            initial_combined.append(combined_disaster)
        
        # Second pass: combine disasters with matching locations
        location_clusters = defaultdict(list)
        
        # Group by location
        for i, disaster in enumerate(initial_combined):
            added_to_cluster = False
            
            # Check existing clusters
            for loc_key in location_clusters.keys():
                if self._are_locations_same(
                    disaster['location'],
                    initial_combined[loc_key]['location']
                ):
                    location_clusters[loc_key].append(i)
                    added_to_cluster = True
                    break
            
            # Create new cluster if no match found
            if not added_to_cluster:
                location_clusters[i].append(i)
        
        # Final combination
        final_combined = []
        
        for cluster_indices in location_clusters.values():
            disasters_to_merge = [initial_combined[i] for i in cluster_indices]
            merged_disaster = self._merge_disaster_data(disasters_to_merge)
            final_combined.append(merged_disaster)
        
        # Create final DataFrame
        if not final_combined:
            return pd.DataFrame(columns=[
                'disaster_type', 'severity', 'location', 'published_at',
                'urls', 'text', 'article_count'
            ])
        
        return pd.DataFrame(final_combined)

    def _get_most_common_value(self, series):
        """
        Get the most common non-null value from a series
        """
        values = series.dropna()
        if values.empty:
            return 'unknown'
        return values.mode().iloc[0]

    def _get_highest_severity(self, severities):
        """
        Get the highest severity from a series of severities
        """
        severity_rank = {'high': 3, 'medium': 2, 'low': 1, 'unknown': 0}
        return max(severities, key=lambda x: severity_rank.get(x, 0)) 