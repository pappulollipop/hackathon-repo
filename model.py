import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

disease_data_file = 'Final_Augmented_dataset_Diseases_and_Symptoms.csv (1).zip'
disease_data = pd.read_csv(disease_data_file)
disease_data.describe()

disease_data.head()

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

#disease encoder
def encode_diseases_complete(df, disease_column='diseases', save_files=True):
    print(f"Original data shape: {df.shape}")
    print(f"Disease column: '{disease_column}'")
    if disease_column not in df.columns:
        raise ValueError(f"Column '{disease_column}' not found in DataFrame")

    encoder = LabelEncoder()
    encoded_df = df.copy()
    unique_diseases = df[disease_column].unique()
    print(f"Found {len(unique_diseases)} unique diseases: {list(unique_diseases)}")

    # Encode diseases
    encoded_df[f'{disease_column}_encoded'] = encoder.fit_transform(encoded_df[disease_column])

    # Create comprehensive mapping dictionaries
    disease_to_number = {disease: int(code) for code, disease in enumerate(encoder.classes_)}
    number_to_disease = {int(code): disease for code, disease in enumerate(encoder.classes_)}

    # Create complete mapping record
    mappings = {
        'disease_to_number': disease_to_number,
        'number_to_disease': number_to_disease,
        'encoder_classes': encoder.classes_.tolist(),
        'total_diseases': len(unique_diseases),
        'encoding_info': {
            'method': 'LabelEncoder',
            'range': f'0 to {len(unique_diseases)-1}',
            'total_records': len(df)
        }
    }

    # Display mapping information
    print("\n" + "="*50)
    print("DISEASE ENCODING MAPPING")
    print("="*50)
    for disease, code in disease_to_number.items():
        print(f"  '{disease}' → {code}")

    # Show distribution
    print("\n" + "="*50)
    print("DISEASE DISTRIBUTION")
    print("="*50)
    distribution = encoded_df[f'{disease_column}_encoded'].value_counts().sort_index()
    for code, count in distribution.items():
        disease_name = number_to_disease[code]
        percentage = (count / len(encoded_df)) * 100
        print(f"  Code {code} ({disease_name}): {count} records ({percentage:.1f}%)")

    # Save files if requested
    if save_files:
        # Save mapping as JSON
        with open('../final/disease_encoding_mapping.json', 'w') as f:
            json.dump(mappings, f, indent=2)

        # Save mapping as CSV
        mapping_df = pd.DataFrame({
            'Disease_Name': list(disease_to_number.keys()),
            'Disease_Code': list(disease_to_number.values())
        })
        mapping_df.to_csv('disease_encoding_mapping.csv', index=False)

        # Save encoded dataframe
        encoded_df.to_csv('data_with_encoded_diseases.csv', index=False)

        print("\n✓ Files saved:")
        print("  - disease_encoding_mapping.json")
        print("  - disease_encoding_mapping.csv")
        print("  - data_with_encoded_diseases.csv")

    return encoded_df, mappings

# ===== GROUPING FUNCTION WITH ENCODED DISEASES =====
def group_by_encoded_diseases(encoded_df, mappings, disease_column='diseases'):
    """
    Group data by encoded diseases and calculate averages
    """
    print("\n" + "="*50)
    print("GROUPING BY ENCODED DISEASES")
    print("="*50)

    # Select only numeric columns for grouping
    numeric_columns = encoded_df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove the encoded disease column from the grouping calculation
    symptom_columns = [col for col in numeric_columns if not col.endswith('_encoded')]

    print(f"Grouping by: {disease_column}_encoded")
    print(f"Calculating averages for: {symptom_columns}")

    # Group by encoded diseases
    grouped_data = encoded_df[symptom_columns].groupby(encoded_df[f'{disease_column}_encoded']).mean().round(3)

    # Create version with disease names for reference
    grouped_with_names = grouped_data.copy()
    grouped_with_names.index = [f"Code_{idx}_({mappings['number_to_disease'][idx]})"
                               for idx in grouped_with_names.index]

    print("\nGrouped data by encoded diseases:")
    print(grouped_data)

    # Save grouped data
    grouped_data.to_csv('grouped_data_by_encoded_diseases.csv')
    grouped_with_names.to_csv('grouped_data_with_disease_names.csv')

    print("\n✓ Grouped data saved:")
    print("  - grouped_data_by_encoded_diseases.csv")
    print("  - grouped_data_with_disease_names.csv")

    return grouped_data, grouped_with_names

# ===== UTILITY FUNCTIONS =====
def load_disease_mapping(json_file='disease_encoding_mapping.json'):
    """Load disease mapping from saved JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def disease_name_to_code(disease_name, mappings):
    """Convert disease name to code"""
    return mappings['disease_to_number'].get(disease_name, None)

def disease_code_to_name(code, mappings):
    """Convert disease code to name"""
    return mappings['number_to_disease'].get(code, None)

def show_complete_mapping_info(mappings):
    """Display complete mapping information"""
    print("="*50)
    print("COMPLETE DISEASE MAPPING INFORMATION")
    print("="*50)
    print(f"Total diseases: {mappings['total_diseases']}")
    print(f"Encoding method: {mappings['encoding_info']['method']}")
    print(f"Code range: {mappings['encoding_info']['range']}")
    print(f"Total records: {mappings['encoding_info']['total_records']}")
    print("\nComplete mapping:")
    for disease, code in mappings['disease_to_number'].items():
        print(f"  '{disease}' ↔ {code}")
df, mappings = encode_diseases_complete(disease_data)
grouped_data, grouped_with_names= group_by_encoded_diseases(df, mappings)
code = disease_name_to_code('panic disorder', mappings)
name = disease_code_to_name(0, mappings)
show_complete_mapping_info(mappings)

df.head()

# Split the data into training and testing sets
X = df.drop(["diseases", 'diseases_encoded'],axis=1)
y = df["diseases_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier  # Changed from Regressor

# Use classifier instead of regressor for better disease prediction
disease_model = DecisionTreeClassifier(
    random_state=1,
    min_samples_split=5,  # Prevent overfitting
    max_depth=15,         # Limit tree depth
    min_samples_leaf=3    # Minimum samples per leaf
)

disease_model.fit(X_train, y_train)
print("✅ Decision Tree Classifier trained successfully!")

final_model = DecisionTreeClassifier(
    random_state=1,
    min_samples_split=5,
    max_depth=15,
    min_samples_leaf=3
)
final_model.fit(X, y)
print("✅ Final model trained with enhanced parameters!")

def get_top_3_disease_predictions(binary_vector, model, mappings):
    """
    Get top 3 disease predictions with confidence percentages

    Returns: List of (disease_name, confidence_percentage)
    """
    # Ensure correct shape
    if len(binary_vector.shape) == 1:
        binary_vector = binary_vector.reshape(1, -1)

    # Get probability predictions for all diseases
    probabilities = model.predict_proba(binary_vector)[0]
    disease_codes = model.classes_

    # Create list of (disease_code, probability) pairs
    disease_prob_pairs = list(zip(disease_codes, probabilities))

    # Sort by probability (highest first)
    disease_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    # Get top 3 with disease names
    top_3_predictions = []
    for i in range(min(3, len(disease_prob_pairs))):
        disease_code, probability = disease_prob_pairs[i]
        disease_name = mappings['number_to_disease'].get(disease_code, f"Unknown_Disease_{disease_code}")
        confidence_percentage = probability * 100
        top_3_predictions.append((disease_name, confidence_percentage))

    return top_3_predictions

def display_top_predictions(predictions):
    """Display predictions in a formatted way"""
    print("\n" + "="*70)
    print("🎯 TOP 3 DISEASE PREDICTIONS")
    print("="*70)

    for rank, (disease, confidence) in enumerate(predictions, 1):
        # Color coding based on confidence
        if confidence > 60:
            status = "🔴 HIGH CONFIDENCE"
        elif confidence > 30:
            status = "🟡 MEDIUM CONFIDENCE"
        else:
            status = "🟢 LOW CONFIDENCE"

        print(f"\n{rank}. {disease}")
        print(f"   Confidence: {confidence:.2f}% {status}")

        # Add recommendation based on confidence
        if confidence > 70:
            print("   💡 Recommendation: Consult a healthcare professional")
        elif confidence > 40:
            print("   💡 Recommendation: Monitor symptoms and consider medical advice")
        else:
            print("   💡 Recommendation: Continue observation")

    print("\n" + "="*70)

# ===== NEW CELL: ENHANCED SYMPTOM MANAGER =====

class SymptomManager:
    """
    Manages dynamic symptom collection and predictions
    """

    def __init__(self, symptom_extractor, model, mappings):
        self.symptom_extractor = symptom_extractor
        self.model = model
        self.mappings = mappings
        self.current_symptoms = set()
        self.binary_vector = np.zeros(len(symptom_extractor.symptoms_list))

    def add_symptoms_from_text(self, text_description):
        """Extract and add symptoms from text description"""
        # Get binary vector from text
        detected_binary = self.symptom_extractor.extract_symptoms(text_description)

        # Find new symptoms
        new_symptoms = []
        for i, (current, detected) in enumerate(zip(self.binary_vector, detected_binary)):
            if current == 0 and detected == 1:
                symptom_name = self.symptom_extractor.symptoms_list[i]
                new_symptoms.append(symptom_name)
                self.current_symptoms.add(symptom_name)

        # Update binary vector (combine existing and new)
        self.binary_vector = np.maximum(self.binary_vector, detected_binary)

        return new_symptoms

    def remove_symptom_by_name(self, symptom_name):
        """Remove a specific symptom by name"""
        if symptom_name in self.symptom_extractor.symptoms_list:
            index = self.symptom_extractor.symptoms_list.index(symptom_name)
            self.binary_vector[index] = 0
            self.current_symptoms.discard(symptom_name)
            return True
        return False

    def get_predictions(self):
        """Get current top-3 predictions"""
        if np.sum(self.binary_vector) == 0:
            return []
        return get_top_3_disease_predictions(self.binary_vector, self.model, self.mappings)

    def show_current_status(self):
        """Display current symptoms and predictions"""
        print(f"\n📋 CURRENT SYMPTOMS ({len(self.current_symptoms)}):")
        if self.current_symptoms:
            for i, symptom in enumerate(sorted(self.current_symptoms), 1):
                print(f"   {i}. {symptom}")
        else:
            print("   (No symptoms currently recorded)")

        # Show predictions if symptoms exist
        predictions = self.get_predictions()
        if predictions:
            display_top_predictions(predictions)
        else:
            print("\n⚠️ Add symptoms to get disease predictions")

    def clear_all_symptoms(self):
        """Clear all current symptoms"""
        self.current_symptoms.clear()
        self.binary_vector = np.zeros(len(self.symptom_extractor.symptoms_list))
        print("✅ All symptoms cleared!")

# ===== NEW CELL: REPLACE YOUR FINAL PREDICTION WORKFLOW =====

symptom_list=list(X.columns)
symptom_list





import re
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class AdvancedSymptomExtractor:
    """
    Advanced Symptom Extraction Model that uses multiple NLP techniques
    to identify symptoms from complex sentences and generate binary vectors.
    """

    def __init__(self, symptoms_list, similarity_threshold=0.6, fuzzy_threshold=75):
        """
        Initialize the symptom extractor

        Args:
            symptoms_list: List of symptom strings to detect
            similarity_threshold: Threshold for TF-IDF similarity (0-1)
            fuzzy_threshold: Threshold for fuzzy matching (0-100)
        """
        self.symptoms_list = symptoms_list
        self.n_symptoms = len(symptoms_list)
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold

        # Initialize TF-IDF vectorizer for semantic similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000
        )

        # Create preprocessed symptoms and anatomical context
        self._create_symptom_data()
        self._create_anatomical_context()

        print(f"✓ SymptomExtractor initialized with {self.n_symptoms} symptoms")

    def _simple_tokenize(self, text):
        """Simple tokenization without external dependencies"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        stop_words = {'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'the', 'is', 'are', 'was', 'were'}
        tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]

        return tokens

    def _create_symptom_data(self):
        """Create processed symptoms and variations"""
        self.processed_symptoms = []
        self.symptom_keywords = []
        self.symptom_variations = {}

        for i, symptom in enumerate(self.symptoms_list):
            # Process symptom
            tokens = self._simple_tokenize(symptom)
            processed = ' '.join(tokens)
            self.processed_symptoms.append(processed)
            self.symptom_keywords.append(tokens)

            # Create variations
            variations = [symptom.lower(), processed]

            # Add medical variations
            if 'pain' in symptom.lower():
                variations.extend([
                    symptom.lower().replace('pain', 'ache'),
                    symptom.lower().replace('pain', 'hurt'),
                    symptom.lower().replace('pain', 'discomfort')
                ])

            if 'difficulty' in symptom.lower():
                variations.extend([
                    symptom.lower().replace('difficulty', 'trouble'),
                    symptom.lower().replace('difficulty', 'problem')
                ])

            self.symptom_variations[i] = list(set(variations))

        # Fit TF-IDF on all symptoms
        all_symptom_text = self.processed_symptoms + [var for vars_list in self.symptom_variations.values() for var in vars_list]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_symptom_text)

    def _create_anatomical_context(self):
        """Create anatomical context for better specificity"""
        self.anatomical_regions = {
            'chest': ['chest', 'breast', 'thorax', 'heart', 'lung'],
            'head': ['head', 'skull', 'brain', 'face', 'forehead'],
            'throat': ['throat', 'neck', 'pharynx', 'larynx'],
            'abdomen': ['abdomen', 'belly', 'stomach', 'gut'],
            'back': ['back', 'spine', 'vertebra'],
            'leg': ['leg', 'thigh', 'knee', 'calf', 'foot', 'ankle'],
            'arm': ['arm', 'elbow', 'wrist', 'hand', 'finger', 'shoulder'],
            'eye': ['eye', 'vision', 'sight', 'pupil'],
            'ear': ['ear', 'hearing', 'auditory'],
            'urinary': ['urine', 'bladder', 'kidney', 'urinary'],
            'reproductive': ['testicle', 'scrotum', 'vagina', 'reproductive']
        }

    def _check_anatomical_context(self, user_tokens, pain_symptom):
        """Check if pain mention has appropriate anatomical context"""
        pain_symptom_lower = pain_symptom.lower()

        # Map pain symptoms to their anatomical regions
        anatomical_map = {
            'chest': ['sharp chest pain', 'chest tightness'],
            'leg': ['leg pain'],
            'head': ['headache'],
            'back': ['back pain'],
            'arm': ['arm pain', 'elbow pain', 'wrist pain', 'hand pain', 'finger pain'],
            'abdomen': ['sharp abdominal pain', 'abdominal pain', 'lower abdominal pain'],
            'reproductive': ['pain in testicles', 'testicular pain'],
            'hip': ['hip pain'],
            'urinary': ['suprapubic pain']
        }

        # Find which anatomical region this pain belongs to
        target_region = None
        for region, symptoms in anatomical_map.items():
            if any(pain_symptom_lower in s or s in pain_symptom_lower for s in symptoms):
                target_region = region
                break

        if target_region is None:
            return False

        # Check if user mentions words related to this anatomical region
        region_words = self.anatomical_regions.get(target_region, [])
        user_has_region_context = bool(user_tokens.intersection(set(region_words)))

        return user_has_region_context

    def extract_symptoms(self, user_input):
        """
        Extract symptoms from user input text using contextual matching

        Args:
            user_input: String containing user's description

        Returns:
            Binary numpy array where 1 indicates symptom present, 0 indicates absent
        """
        binary_vector = np.zeros(self.n_symptoms, dtype=int)
        user_lower = user_input.lower()
        user_tokens = set(self._simple_tokenize(user_input))

        for i, symptom in enumerate(self.symptoms_list):
            symptom_lower = symptom.lower()

            # For pain symptoms, check anatomical context
            if 'pain' in symptom_lower:
                if self._check_anatomical_context(user_tokens, symptom):
                    binary_vector[i] = 1
                    continue

            # 1. Exact phrase matching (high confidence)
            if symptom_lower in user_lower:
                binary_vector[i] = 1
                continue

            # 2. High fuzzy match
            if fuzz.partial_ratio(user_lower, symptom_lower) >= 85:
                binary_vector[i] = 1
                continue

            # 3. Strong keyword overlap for multi-word symptoms
            symptom_tokens = set(self.symptom_keywords[i])
            if len(symptom_tokens) > 1:
                overlap = len(user_tokens.intersection(symptom_tokens))
                if overlap >= len(symptom_tokens) * 0.8:  # 80% of keywords must match
                    binary_vector[i] = 1
                    continue

            # 4. Single word exact match for single-word symptoms
            elif len(symptom_tokens) == 1:
                if symptom_tokens.issubset(user_tokens):
                    binary_vector[i] = 1

        return binary_vector

    def get_detected_symptoms(self, user_input):
        """
        Get list of detected symptoms with their confidence scores

        Args:
            user_input: String containing user's description

        Returns:
            tuple: (list of detected symptoms with metadata, binary vector)
        """
        binary_vector = self.extract_symptoms(user_input)
        detected_symptoms = []

        for i, is_present in enumerate(binary_vector):
            if is_present:
                detected_symptoms.append({
                    'index': i,
                    'symptom': self.symptoms_list[i],
                    'confidence': self._calculate_confidence(user_input, i)
                })

        return detected_symptoms, binary_vector

    def _calculate_confidence(self, user_input, symptom_index):
        """Calculate confidence score for detected symptom"""
        symptom = self.symptoms_list[symptom_index]

        # Fuzzy matching confidence
        fuzzy_score = fuzz.partial_ratio(user_input.lower(), symptom.lower()) / 100.0

        # Keyword matching confidence
        user_tokens = set(self._simple_tokenize(user_input))
        symptom_tokens = set(self.symptom_keywords[symptom_index])
        keyword_score = len(user_tokens.intersection(symptom_tokens)) / len(symptom_tokens)

        # TF-IDF confidence
        user_processed = ' '.join(self._simple_tokenize(user_input))
        user_tfidf = self.tfidf_vectorizer.transform([user_processed])
        symptom_tfidf = self.tfidf_vectorizer.transform([self.processed_symptoms[symptom_index]])
        tfidf_score = cosine_similarity(user_tfidf, symptom_tfidf)[0][0]

        # Combine scores (weighted average)
        confidence = (0.3 * fuzzy_score + 0.4 * keyword_score + 0.3 * tfidf_score)
        return min(confidence, 1.0)  # Cap at 1.0

# ===== NEW CELL: REPLACE YOUR FINAL PREDICTION WORKFLOW =====

# Initialize the enhanced symptom manager
symptom_detector = AdvancedSymptomExtractor(symptom_list) # Initialize the symptom_detector
symptom_manager = SymptomManager(symptom_detector, final_model, mappings)

def enhanced_disease_consultation():
    """
    Interactive disease consultation system
    """
    print("\n" + "="*80)
    print("🏥 ENHANCED DISEASE PREDICTION SYSTEM")
    print("="*80)
    print("This system provides TOP 3 disease predictions with confidence levels")
    print("You can add symptoms progressively and see updated predictions")
    print("="*80)

    while True:
        print("\n🔹 MAIN MENU:")
        print("1. Add symptoms (describe in natural language)")
        print("2. Remove a specific symptom")
        print("3. View current symptoms and predictions")
        print("4. Clear all symptoms and start over")
        print("5. Exit system")

        try:
            choice = input("\n👉 Enter your choice (1-5): ").strip()

            if choice == '1':
                description = input("\n💬 Describe your symptoms: ")
                new_symptoms = symptom_manager.add_symptoms_from_text(description)

                if new_symptoms:
                    print(f"\n✅ Detected and added {len(new_symptoms)} new symptoms:")
                    for symptom in new_symptoms:
                        print(f"   • {symptom}")

                    # Show updated status
                    symptom_manager.show_current_status()
                else:
                    print("\n⚠️ No new symptoms detected from your description")
                    print("Try being more specific or use medical terms")

            elif choice == '2':
                if symptom_manager.current_symptoms:
                    print("\n📋 Current symptoms:")
                    symptoms_list = sorted(list(symptom_manager.current_symptoms))
                    for i, symptom in enumerate(symptoms_list, 1):
                        print(f"   {i}. {symptom}")

                    try:
                        selection = int(input("\nEnter number to remove: ")) - 1
                        if 0 <= selection < len(symptoms_list):
                            removed_symptom = symptoms_list[selection]
                            symptom_manager.remove_symptom_by_name(removed_symptom)
                            print(f"\n✅ Removed: {removed_symptom}")
                            symptom_manager.show_current_status()
                        else:
                            print("\n❌ Invalid selection")
                    except ValueError:
                        print("\n❌ Please enter a valid number")
                else:
                    print("\n📋 No symptoms to remove")

            elif choice == '3':
                symptom_manager.show_current_status()

            elif choice == '4':
                symptom_manager.clear_all_symptoms()

            elif choice == '5':
                print("\n👋 Thank you for using the Enhanced Disease Prediction System!")
                print("💡 Remember: This tool is for informational purposes only.")
                print("Always consult healthcare professionals for proper diagnosis.")
                break

            else:
                print("\n❌ Invalid choice. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n\n👋 Session terminated. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

print("\n🚀 Starting Enhanced Disease Prediction System...")

# Option 1: Quick single prediction (similar to your old method)
def quick_prediction_demo():
    user_input = input('Please describe your symptoms: ')
    binary_vector = symptom_detector.extract_symptoms(user_input)

    # Get top 3 predictions instead of just one
    top_predictions = get_top_3_disease_predictions(binary_vector, final_model, mappings)

    print("\n🎯 PREDICTION RESULTS:")
    display_top_predictions(top_predictions)

# Option 2: Full interactive system
print("\nChoose your preferred mode:")
print("1. Quick prediction (like your original system)")
print("2. Enhanced interactive system (recommended)")

mode = input("\nEnter choice (1 or 2): ").strip()

if mode == '1':
    quick_prediction_demo()
elif mode == '2':
    enhanced_disease_consultation()
else:
    print("Invalid choice. Starting enhanced system...")
    enhanced_disease_consultation()

pickle.dump(final_model, open("final_model.pkl" , "wb"))