import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

class TextProcessor:
    def __init__(self):
        # Download all required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
            
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy English model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Preprocess the text by cleaning, tokenizing, and lemmatizing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_location(self, text):
        """
        Extract location entities using spaCy NER
        """
        doc = self.nlp(text)
        locations = []
        
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                locations.append(ent.text)
                
        return locations
    
    def extract_disaster_info(self, text):
        """
        Extract disaster type and severity using rule-based approach
        """
        doc = self.nlp(text.lower())
        
        # Define disaster types and severity indicators
        disaster_types = {
            'earthquake': ['earthquake', 'seismic', 'tremor'],
            'flood': ['flood', 'flooding', 'deluge'],
            'hurricane': ['hurricane', 'cyclone', 'typhoon'],
            'wildfire': ['wildfire', 'fire', 'blaze'],
            'tornado': ['tornado', 'twister'],
            'landslide': ['landslide', 'mudslide'],
            'tsunami': ['tsunami', 'tidal wave']
        }
        
        severity_indicators = {
            'high': ['devastating', 'severe', 'massive', 'major', 'catastrophic'],
            'medium': ['moderate', 'significant', 'considerable'],
            'low': ['minor', 'small', 'light']
        }
        
        # Find disaster type
        disaster_type = None
        for dtype, keywords in disaster_types.items():
            if any(keyword in text.lower() for keyword in keywords):
                disaster_type = dtype
                break
        
        # Determine severity
        severity = 'unknown'
        for level, indicators in severity_indicators.items():
            if any(indicator in text.lower() for indicator in indicators):
                severity = level
                break
                
        return disaster_type, severity 