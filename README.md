# Disease Prediction System - Flask Web Application

A comprehensive web-based disease prediction system that uses machine learning to analyze symptoms and provide potential disease predictions with confidence scores.

## 🚀 Project Overview

This Flask web application integrates a trained Decision Tree Classifier with an advanced symptom extraction system to provide real-time disease predictions. The system uses natural language processing to understand user symptom descriptions and provides top-3 disease predictions with confidence scores and medical recommendations.

## ✨ Features

- **Intelligent Symptom Analysis**: Advanced NLP-based symptom extraction from natural language descriptions
- **Real-time Predictions**: Instant disease predictions using trained machine learning model
- **Interactive Web Interface**: User-friendly frontend with real-time updates
- **Symptom Management**: Add, remove, and track symptoms dynamically
- **Confidence Scoring**: Provides confidence percentages for each prediction
- **Medical Recommendations**: Contextual advice based on prediction confidence
- **REST API**: Complete API endpoints for programmatic access

## 🛠️ Technical Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn (Decision Tree Classifier)
- **NLP**: TF-IDF Vectorization, Fuzzy String Matching
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy
- **Cross-Origin**: Flask-CORS

## 📁 Project Structure

```
disease-prediction-system/
├── app.py                                                    # Main Flask application
├── Final_Augmented_dataset_Diseases_and_Symptoms.csv        # Dataset
├── requirements.txt                                          # Python dependencies
├── templates/
│   └── index.html                                           # Frontend interface
├── static/                                                  # Static files (if any)
└── README.md                                               # This file
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone or Download Project

```bash
# If using git
git clone <your-repository-url>
cd disease-prediction-system

# Or download and extract the project files
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Dataset

Ensure your dataset `Final_Augmented_dataset_Diseases_and_Symptoms.csv` is in the project root directory.

### Step 4: Run the Application

```bash
python app.py
```

### Step 5: Access the Application

Open your browser and navigate to: `http://localhost:5000`

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
Flask==2.3.3
Flask-CORS==4.0.0
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
fuzzywuzzy==0.18.0
python-Levenshtein==0.21.1
```

## 🎯 Usage Instructions

### Adding Symptoms

1. Navigate to the "Add Symptoms" section
2. Enter your symptom description in natural language
   - Example: "I have severe headache and fever"
   - Example: "Experiencing chest pain and difficulty breathing"
3. Click "Analyze Symptoms"
4. The system will extract symptoms and provide predictions

### Viewing Status

1. Click "View Status" to see:
   - Current symptoms list
   - Top-3 disease predictions with confidence scores
   - Medical recommendations

### Managing Symptoms

1. **Remove Symptoms**: Use "Remove Symptom" section to delete specific symptoms
2. **Clear All**: Reset all symptoms and start over

### Understanding Predictions

- **High Confidence (>70%)**: Consult healthcare professional
- **Medium Confidence (40-70%)**: Monitor symptoms, consider medical advice
- **Low Confidence (<40%)**: Continue observation

## 🔗 API Endpoints

### POST /api/analyze_symptoms
Analyze user symptom description and return predictions.

**Request:**
```json
{
  "symptoms": "I have headache and fever"
}
```

**Response:**
```json
{
  "success": true,
  "new_symptoms": ["headache", "fever"],
  "predictions": [
    {
      "name": "flu",
      "confidence": 85.5,
      "recommendation": "Consult a healthcare professional"
    }
  ],
  "total_symptoms": 2
}
```

### POST /api/remove_symptom
Remove a specific symptom by name.

**Request:**
```json
{
  "symptom_name": "headache"
}
```

### GET /api/get_status
Get current symptoms and predictions.

**Response:**
```json
{
  "success": true,
  "symptoms": ["fever", "cough"],
  "predictions": [...],
  "total_symptoms": 2
}
```

### POST /api/clear_all
Clear all symptoms and reset the system.

## 🧠 Machine Learning Model

### Model Details

- **Algorithm**: Decision Tree Classifier
- **Parameters**: 
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 3
  - random_state: 1

### Symptom Extraction

The `AdvancedSymptomExtractor` uses:
- **TF-IDF Vectorization** for text similarity
- **Fuzzy String Matching** for symptom variations
- **Anatomical Context Analysis** for pain-related symptoms
- **Keyword Overlap Detection** for multi-word symptoms

### Data Processing

- **Dataset**: Augmented disease and symptom dataset
- **Encoding**: Label encoding for disease classification
- **Features**: Binary symptom presence vectors
- **Target**: Encoded disease labels

## 🔧 Core Classes

### AdvancedSymptomExtractor
Handles natural language symptom extraction with:
- Text preprocessing and tokenization
- Similarity matching using TF-IDF
- Anatomical context validation
- Fuzzy string matching

### SymptomManager
Manages symptom collection and predictions:
- Dynamic symptom addition/removal
- Real-time prediction updates
- Binary vector maintenance
- State management

## 🐛 Troubleshooting

### Common Issues

#### "index.html not found"
**Solution**: Ensure `index.html` is in the `templates/` directory.

#### "CORS policy" errors
**Solution**: Verify `Flask-CORS` is installed and `CORS(app)` is added to your Flask app.

#### Buttons not working
**Solution**: 
- Hard refresh browser (Ctrl+Shift+R)
- Check browser console for JavaScript errors
- Verify Flask server is running on port 5000

#### "No symptoms detected"
**Solution**: 
- Use more specific medical terminology
- Check if your dataset loaded correctly
- Verify symptom extraction is working

#### Model initialization errors
**Solution**: 
- Ensure dataset file path is correct
- Check dataset format and columns
- Verify all dependencies are installed

### Debugging Steps

1. **Check Flask logs** for server-side errors
2. **Open browser Developer Tools** (F12) → Network tab
3. **Verify API calls** are being made to correct endpoints
4. **Check console** for JavaScript errors
5. **Test API endpoints** individually using tools like Postman

## 🔒 Security Considerations

- **Input Validation**: All user inputs are sanitized
- **Error Handling**: Comprehensive error handling prevents crashes
- **CORS**: Properly configured for web security
- **No Sensitive Data**: No personal health information is stored

## 🚀 Future Enhancements

### Potential Improvements

1. **Database Integration**: Store symptom history and predictions
2. **User Authentication**: Personal health tracking
3. **Enhanced NLP**: Better symptom extraction with advanced NLP models
4. **Mobile Responsiveness**: Improved mobile interface
5. **Caching**: Redis caching for better performance
6. **Logging**: Comprehensive logging system
7. **Testing**: Unit tests and integration tests
8. **Deployment**: Docker containerization for easy deployment

### Model Improvements

1. **Ensemble Methods**: Combine multiple algorithms
2. **Feature Engineering**: Advanced symptom relationships
3. **Model Validation**: Cross-validation and performance metrics
4. **Real-time Learning**: Model updates with new data
5. **Explainability**: Model interpretation features

## 📊 Performance Metrics

- **Response Time**: < 500ms for symptom analysis
- **Model Accuracy**: Based on training dataset performance
- **Symptom Detection**: Advanced NLP with high precision
- **Scalability**: Handles multiple concurrent users

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- naman khandelwal, sarabjeet singh , shreyash pandey, aman jain- Initial work and development

## 🙏 Acknowledgments

- scikit-learn for machine learning tools
- Flask for web framework
- TF-IDF for text processing
- FuzzyWuzzy for string matching

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the API documentation
3. Check browser console for errors
4. Verify Flask server logs

---

**⚠️ Medical Disclaimer**: This system is for educational purposes only. Always consult healthcare professionals for medical advice. Do not use this system as a substitute for professional medical diagnosis or treatment.

**Last Updated**: July 2025
