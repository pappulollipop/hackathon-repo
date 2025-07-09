from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import json
import re
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

app = Flask(__name__)
CORS(app)

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

warnings.filterwarnings('ignore')

  # Enable CORS for all routes

# Global variables to store model and components
final_model = None
mappings = None
symptom_detector = None
symptom_manager = None


class AdvancedSymptomExtractor:
    """Advanced Symptom Extraction Model"""

    def _init_(self, symptoms_list, similarity_threshold=0.6, fuzzy_threshold=75):
        self.symptoms_list = symptoms_list
        self.n_symptoms = len(symptoms_list)
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000
        )

        self._create_symptom_data()
        self._create_anatomical_context()
        print(f"✓ SymptomExtractor initialized with {self.n_symptoms} symptoms")

    def _simple_tokenize(self, text):
        """Simple tokenization without external dependencies"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        stop_words = {'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'the', 'is',
                      'are', 'was', 'were'}
        tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]
        return tokens

    def _create_symptom_data(self):
        """Create processed symptoms and variations"""
        self.processed_symptoms = []
        self.symptom_keywords = []
        self.symptom_variations = {}

        for i, symptom in enumerate(self.symptoms_list):
            tokens = self._simple_tokenize(symptom)
            processed = ' '.join(tokens)
            self.processed_symptoms.append(processed)
            self.symptom_keywords.append(tokens)

            # Create variations
            variations = [symptom.lower(), processed]
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
        all_symptom_text = self.processed_symptoms + [var for vars_list in self.symptom_variations.values() for var in
                                                      vars_list]
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

        target_region = None
        for region, symptoms in anatomical_map.items():
            if any(pain_symptom_lower in s or s in pain_symptom_lower for s in symptoms):
                target_region = region
                break

        if target_region is None:
            return False

        region_words = self.anatomical_regions.get(target_region, [])
        user_has_region_context = bool(user_tokens.intersection(set(region_words)))
        return user_has_region_context

    def extract_symptoms(self, user_input):
        """Extract symptoms from user input text"""
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

            # Exact phrase matching
            if symptom_lower in user_lower:
                binary_vector[i] = 1
                continue

            # High fuzzy match
            if fuzz.partial_ratio(user_lower, symptom_lower) >= 85:
                binary_vector[i] = 1
                continue

            # Strong keyword overlap for multi-word symptoms
            symptom_tokens = set(self.symptom_keywords[i])
            if len(symptom_tokens) > 1:
                overlap = len(user_tokens.intersection(symptom_tokens))
                if overlap >= len(symptom_tokens) * 0.8:
                    binary_vector[i] = 1
                    continue
            # Single word exact match
            elif len(symptom_tokens) == 1:
                if symptom_tokens.issubset(user_tokens):
                    binary_vector[i] = 1

        return binary_vector

    def get_detected_symptoms(self, user_input):
        """Get list of detected symptoms with their names"""
        binary_vector = self.extract_symptoms(user_input)
        detected_symptoms = []

        for i, is_present in enumerate(binary_vector):
            if is_present:
                detected_symptoms.append(self.symptoms_list[i])

        return detected_symptoms, binary_vector


class SymptomManager:
    """Manages dynamic symptom collection and predictions"""

    def _init_(self, symptom_extractor, model, mappings):
        self.symptom_extractor = symptom_extractor
        self.model = model
        self.mappings = mappings
        self.current_symptoms = set()
        self.binary_vector = np.zeros(len(symptom_extractor.symptoms_list))

    def add_symptoms_from_text(self, text_description):
        """Extract and add symptoms from text description"""
        detected_binary = self.symptom_extractor.extract_symptoms(text_description)

        new_symptoms = []
        for i, (current, detected) in enumerate(zip(self.binary_vector, detected_binary)):
            if current == 0 and detected == 1:
                symptom_name = self.symptom_extractor.symptoms_list[i]
                new_symptoms.append(symptom_name)
                self.current_symptoms.add(symptom_name)

        # Update binary vector
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

    def clear_all_symptoms(self):
        """Clear all current symptoms"""
        self.current_symptoms.clear()
        self.binary_vector = np.zeros(len(self.symptom_extractor.symptoms_list))

    def get_current_symptoms(self):
        """Get current symptoms as a list"""
        return list(self.current_symptoms)


def encode_diseases_complete(df, disease_column='diseases'):
    """Encode diseases using LabelEncoder"""
    encoder = LabelEncoder()
    encoded_df = df.copy()

    # Encode diseases
    encoded_df[f'{disease_column}_encoded'] = encoder.fit_transform(encoded_df[disease_column])

    # Create mapping dictionaries
    disease_to_number = {disease: int(code) for code, disease in enumerate(encoder.classes_)}
    number_to_disease = {int(code): disease for code, disease in enumerate(encoder.classes_)}

    mappings = {
        'disease_to_number': disease_to_number,
        'number_to_disease': number_to_disease,
        'encoder_classes': encoder.classes_.tolist(),
        'total_diseases': len(encoder.classes_)
    }

    return encoded_df, mappings


def get_top_3_disease_predictions(binary_vector, model, mappings):
    """Get top 3 disease predictions with confidence percentages"""
    if len(binary_vector.shape) == 1:
        binary_vector = binary_vector.reshape(1, -1)

    # Get probability predictions
    probabilities = model.predict_proba(binary_vector)[0]
    disease_codes = model.classes_

    # Create list of (disease_code, probability) pairs
    disease_prob_pairs = list(zip(disease_codes, probabilities))
    disease_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    # Get top 3 with disease names
    top_3_predictions = []
    for i in range(min(3, len(disease_prob_pairs))):
        disease_code, probability = disease_prob_pairs[i]
        disease_name = mappings['number_to_disease'].get(disease_code, f"Unknown_Disease_{disease_code}")
        confidence_percentage = probability * 100

        # Add recommendation based on confidence
        if confidence_percentage > 70:
            recommendation = "Consult a healthcare professional"
        elif confidence_percentage > 40:
            recommendation = "Monitor symptoms and consider medical advice"
        else:
            recommendation = "Continue observation"

        top_3_predictions.append({
            'name': disease_name,
            'confidence': confidence_percentage,
            'recommendation': recommendation
        })

    return top_3_predictions

def initialize_model():
        """Initialize the disease prediction model with your actual dataset"""
        global final_model, mappings, symptom_detector, symptom_manager

        try:
            print("🔄 Loading actual dataset...")

            # Load your actual dataset
            disease_data_file = 'Final_Augmented_dataset_Diseases_and_Symptoms.csv (1).zip'
            disease_data = pd.read_csv(disease_data_file)

            # Encode diseases using your actual function
            df, mappings = encode_diseases_complete(disease_data)

            # Prepare features and target (your actual approach)
            X = df.drop(["diseases", 'diseases_encoded'], axis=1)
            y = df["diseases_encoded"]

            # Get actual symptom list from your dataset
            symptom_list = list(X.columns)

            # Train your actual model
            final_model = DecisionTreeClassifier(
                random_state=1,
                min_samples_split=5,
                max_depth=15,
                min_samples_leaf=3
            )
            final_model.fit(X, y)

            # Initialize with your actual classes
            symptom_detector = AdvancedSymptomExtractor()
            symptom_manager = SymptomManager()

            return True

        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            return False

        # Initialize model - replace with your actual model training
        final_model = DecisionTreeClassifier(
            random_state=1,
            min_samples_split=5,
            max_depth=15,
            min_samples_leaf=3
        )

        # You need to train your model here with actual data
        # final_model.fit(X, y)

        # Initialize symptom detector and manager
        symptom_detector = AdvancedSymptomExtractor(symptom_list)
        symptom_manager = SymptomManager(symptom_detector, final_model, mappings)

        print("✅ Model initialized successfully!")
        return True




# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/analyze_symptoms', methods=['POST'])
def analyze_symptoms():
    """Analyze symptoms from user input"""
    try:
        data = request.get_json()
        user_input = data.get('symptoms', '')

        if not user_input:
            return jsonify({'error': 'No symptoms provided'}), 400

        # Extract symptoms
        new_symptoms = symptom_manager.add_symptoms_from_text(user_input)

        # Get predictions
        predictions = symptom_manager.get_predictions()

        return jsonify({
            'success': True,
            'new_symptoms': new_symptoms,
            'predictions': predictions,
            'total_symptoms': len(symptom_manager.current_symptoms)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/remove_symptom', methods=['POST'])
def remove_symptom():
    """Remove a specific symptom"""
    try:
        data = request.get_json()
        symptom_name = data.get('symptom_name', '')

        if not symptom_name:
            return jsonify({'error': 'No symptom name provided'}), 400

        success = symptom_manager.remove_symptom_by_name(symptom_name)

        if success:
            predictions = symptom_manager.get_predictions()
            return jsonify({
                'success': True,
                'predictions': predictions,
                'total_symptoms': len(symptom_manager.current_symptoms)
            })
        else:
            return jsonify({'error': 'Symptom not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_status', methods=['GET'])
def get_status():
    """Get current symptoms and predictions"""
    try:
        current_symptoms = symptom_manager.get_current_symptoms()
        predictions = symptom_manager.get_predictions()

        return jsonify({
            'success': True,
            'symptoms': current_symptoms,
            'predictions': predictions,
            'total_symptoms': len(current_symptoms)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_all', methods=['POST'])
def clear_all():
    """Clear all symptoms"""
    try:
        symptom_manager.clear_all_symptoms()

        return jsonify({
            'success': True,
            'message': 'All symptoms cleared successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def encode_diseases_complete(df, disease_column='diseases'):
    """Encode diseases using LabelEncoder"""
    encoder = LabelEncoder()
    encoded_df = df.copy()

    # Encode diseases
    encoded_df[f'{disease_column}_encoded'] = encoder.fit_transform(encoded_df[disease_column])

    # Create mapping dictionaries
    disease_to_number = {disease: int(code) for code, disease in enumerate(encoder.classes_)}
    number_to_disease = {int(code): disease for code, disease in enumerate(encoder.classes_)}

    mappings = {
        'disease_to_number': disease_to_number,
        'number_to_disease': number_to_disease,
        'encoder_classes': encoder.classes_.tolist(),
        'total_diseases': len(encoder.classes_)
    }

    return encoded_df, mappings


def get_top_3_disease_predictions(binary_vector, model, mappings):
    """Get top 3 disease predictions with confidence percentages"""
    if len(binary_vector.shape) == 1:
        binary_vector = binary_vector.reshape(1, -1)

    # Get probability predictions
    probabilities = model.predict_proba(binary_vector)[0]
    disease_codes = model.classes_

    # Create list of (disease_code, probability) pairs
    disease_prob_pairs = list(zip(disease_codes, probabilities))
    disease_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    # Get top 3 with disease names
    top_3_predictions = []
    for i in range(min(3, len(disease_prob_pairs))):
        disease_code, probability = disease_prob_pairs[i]
        disease_name = mappings['number_to_disease'].get(disease_code, f"Unknown_Disease_{disease_code}")
        confidence_percentage = probability * 100

        # Add recommendation based on confidence
        if confidence_percentage > 70:
            recommendation = "Consult a healthcare professional"
        elif confidence_percentage > 40:
            recommendation = "Monitor symptoms and consider medical advice"
        else:
            recommendation = "Continue observation"

        top_3_predictions.append({
            'name': disease_name,
            'confidence': confidence_percentage,
            'recommendation': recommendation
        })

    return top_3_predictions


def encode_diseases_complete(df, disease_column='diseases'):
    """Encode diseases using LabelEncoder"""
    encoder = LabelEncoder()
    encoded_df = df.copy()

    # Encode diseases
    encoded_df[f'{disease_column}_encoded'] = encoder.fit_transform(encoded_df[disease_column])

    # Create mapping dictionaries
    disease_to_number = {disease: int(code) for code, disease in enumerate(encoder.classes_)}
    number_to_disease = {int(code): disease for code, disease in enumerate(encoder.classes_)}

    mappings = {
        'disease_to_number': disease_to_number,
        'number_to_disease': number_to_disease,
        'encoder_classes': encoder.classes_.tolist(),
        'total_diseases': len(encoder.classes_)
    }

    return encoded_df, mappings


def get_top_3_disease_predictions(binary_vector, model, mappings):
    """Get top 3 disease predictions with confidence percentages"""
    if len(binary_vector.shape) == 1:
        binary_vector = binary_vector.reshape(1, -1)

    # Get probability predictions
    probabilities = model.predict_proba(binary_vector)[0]
    disease_codes = model.classes_

    # Create list of (disease_code, probability) pairs
    disease_prob_pairs = list(zip(disease_codes, probabilities))
    disease_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    # Get top 3 with disease names
    top_3_predictions = []
    for i in range(min(3, len(disease_prob_pairs))):
        disease_code, probability = disease_prob_pairs[i]
        disease_name = mappings['number_to_disease'].get(disease_code, f"Unknown_Disease_{disease_code}")
        confidence_percentage = probability * 100

        # Add recommendation based on confidence
        if confidence_percentage > 70:
            recommendation = "Consult a healthcare professional"
        elif confidence_percentage > 40:
            recommendation = "Monitor symptoms and consider medical advice"
        else:
            recommendation = "Continue observation"

        top_3_predictions.append({
            'name': disease_name,
            'confidence': confidence_percentage,
            'recommendation': recommendation
        })

    return top_3_predictions


if __name__ == '__main__':
    # Initialize the model
    if initialize_model():
        print("🚀 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize model. Please check your dataset and try again.")