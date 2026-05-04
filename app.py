from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os
import warnings
import json
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    print("⚠️  SHAP not available. Explainable AI features will be disabled.")

# Load environment variables (robust to missing/locked .env)
try:
    load_dotenv()
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}. Continuing with default environment variables.")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

app = Flask(__name__)
app.json_encoder = NumpyEncoder  # Use custom JSON encoder

# Global variables for model components
model = None
scaler = None
le = None
model_loaded = False
shap_explainer = None
feature_names = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

# Clinical reference ranges for context in SHAP explanations
CLINICAL_RANGES = {
    'Age': {'normal': (18, 35), 'unit': 'years', 'description': 'Maternal age'},
    'SystolicBP': {'normal': (90, 120), 'unit': 'mmHg', 'description': 'Systolic blood pressure'},
    'DiastolicBP': {'normal': (60, 80), 'unit': 'mmHg', 'description': 'Diastolic blood pressure'},
    'BS': {'normal': (3.9, 5.5), 'unit': 'mmol/L', 'description': 'Blood glucose level'},
    'BodyTemp': {'normal': (36.1, 37.2), 'unit': '°C', 'description': 'Body temperature'},
    'HeartRate': {'normal': (60, 100), 'unit': 'bpm', 'description': 'Resting heart rate'}
}

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'prenatalguard'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': int(os.getenv('DB_PORT', 3306))
}

def get_db_connection():
    """Create and return a database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

def init_database():
    """Initialize database and create tables if they don't exist"""
    connection = get_db_connection()
    if not connection:
        print("⚠️ Could not connect to database. Running without database storage.")
        return False
    
    try:
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Create patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100),
                age INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id VARCHAR(50) NOT NULL,
                prediction_date DATE NOT NULL,
                prediction_month INT NOT NULL,
                prediction_year INT NOT NULL,
                age FLOAT,
                systolic_bp FLOAT,
                diastolic_bp FLOAT,
                blood_sugar FLOAT,
                body_temp FLOAT,
                heart_rate FLOAT,
                risk_level VARCHAR(20) NOT NULL,
                confidence FLOAT,
                recommendations JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                INDEX idx_patient_date (patient_id, prediction_date),
                INDEX idx_month_year (prediction_month, prediction_year)
            )
        """)
        
        connection.commit()
        print("✅ Database initialized successfully")
        return True
        
    except Error as e:
        print(f"❌ Error initializing database: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def save_prediction_to_db(patient_id, health_data, risk_level, confidence, recommendations, pregnancy_month=None):
    """Save prediction results to database
    
    Args:
        patient_id: Patient identifier
        health_data: Dictionary with health metrics
        risk_level: Predicted risk level
        confidence: Prediction confidence
        recommendations: Recommendations dictionary
        pregnancy_month: Pregnancy month (1-11), defaults to None (will use calendar month for backward compatibility)
    """
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        # Insert or update patient
        cursor.execute("""
            INSERT INTO patients (patient_id, name, age)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                age = VALUES(age),
                updated_at = CURRENT_TIMESTAMP
        """, (patient_id, health_data.get('name', ''), health_data.get('Age', 0)))
        
        # Get current date info
        now = datetime.now()
        prediction_date = now.date()
        
        # Use pregnancy_month (1-11) if provided, otherwise use calendar month for backward compatibility
        if pregnancy_month is not None:
            prediction_month = pregnancy_month  # Pregnancy month (1-11)
        else:
            prediction_month = now.month  # Calendar month (1-12) for backward compatibility
        
        prediction_year = now.year
        
        # Insert prediction
        cursor.execute("""
            INSERT INTO predictions (
                patient_id, prediction_date, prediction_month, prediction_year,
                age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate,
                risk_level, confidence, recommendations
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            patient_id, prediction_date, prediction_month, prediction_year,
            health_data.get('Age'),
            health_data.get('SystolicBP'),
            health_data.get('DiastolicBP'),
            health_data.get('BS'),
            health_data.get('BodyTemp'),
            health_data.get('HeartRate'),
            risk_level,
            confidence,
            json.dumps(recommendations)
        ))
        
        connection.commit()
        print(f"✅ Prediction saved to database for patient {patient_id}")
        return True
        
    except Error as e:
        print(f"❌ Error saving prediction to database: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_monthly_predictions(patient_id, pregnancy_month=None, year=None):
    """Fetch monthly predictions for a patient
    
    Args:
        patient_id: Patient identifier
        pregnancy_month: Pregnancy month (1-11) to filter by, or None for all
        year: Calendar year to filter by (optional, for backward compatibility)
    """
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        if pregnancy_month is not None:
            # Get specific pregnancy month (1-11)
            cursor.execute("""
                SELECT * FROM predictions
                WHERE patient_id = %s AND prediction_month = %s
                ORDER BY prediction_date DESC
            """, (patient_id, pregnancy_month))
        elif year:
            # Get by calendar year (for backward compatibility)
            cursor.execute("""
                SELECT * FROM predictions
                WHERE patient_id = %s AND prediction_year = %s
                ORDER BY prediction_month DESC, prediction_date DESC
            """, (patient_id, year))
        else:
            # Get all predictions
            cursor.execute("""
                SELECT * FROM predictions
                WHERE patient_id = %s
                ORDER BY prediction_month ASC, prediction_date DESC
            """, (patient_id,))
        
        results = cursor.fetchall()
        
        # Parse JSON recommendations
        for result in results:
            if result.get('recommendations'):
                try:
                    result['recommendations'] = json.loads(result['recommendations'])
                except:
                    result['recommendations'] = {}
        
        return results
        
    except Error as e:
        print(f"❌ Error fetching predictions: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_all_patients():
    """Get list of all patients"""
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT patient_id, name, age FROM patients ORDER BY created_at DESC")
        return cursor.fetchall()
    except Error as e:
        print(f"❌ Error fetching patients: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def load_models():
    """Load ML models with comprehensive error handling"""
    global model, scaler, le, model_loaded, shap_explainer
    
    try:
        print("🔄 Loading XGBoost model...")
        
        # Load the pre-trained components with error handling
        try:
            print("📁 Loading model files...")
            
            # Load model with specific encoding to handle XGBoost warnings
            with open('xgb_model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully")
            
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded successfully")
            
            with open('label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
            print("✅ Label encoder loaded successfully")
            
            # Initialize SHAP explainer
            if SHAP_AVAILABLE:
                try:
                    print("🔍 Initializing SHAP explainer...")
                    shap_explainer = shap.TreeExplainer(model)
                    print("✅ SHAP explainer initialized successfully")
                except Exception as e:
                    print(f"⚠️  Warning: Could not initialize SHAP explainer: {e}")
                    print("   Predictions will work but SHAP explanations will not be available")
                    shap_explainer = None
            else:
                print("⚠️  SHAP not available - Explainable AI features disabled")
                shap_explainer = None
            
            model_loaded = True
            print(f"📊 Model info: {len(le.classes_)} risk levels - {list(le.classes_)}")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model file not found: {e}")
            print("💡 Please run XGboost_fixed.py first to train the model")
            return False
        except Exception as e:
            print(f"❌ Error loading model files: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error during model loading: {e}")
        return False

def validate_health_data(data):
    """Validate medical data ranges with realistic limits"""
    try:
        checks = {
            'Age': (10, 70, "Age should be between 10 and 70 years"),
            'SystolicBP': (60, 200, "Systolic BP should be between 60 and 200 mmHg"),
            'DiastolicBP': (30, 120, "Diastolic BP should be between 30 and 120 mmHg"),
            'BS': (3, 20, "Blood sugar should be between 3 and 20 mmol/L"),
            'BodyTemp': (35, 42, "Body temperature should be between 35°C and 42°C"),
            'HeartRate': (40, 180, "Heart rate should be between 40 and 180 bpm")
        }
        
        validated_data = {}
        for feature, (min_val, max_val, error_msg) in checks.items():
            value = float(data.get(feature, 0))
            if not (min_val <= value <= max_val):
                return False, f"{error_msg} (got {value})"
            validated_data[feature] = value
            
        return True, validated_data
        
    except ValueError as e:
        return False, f"Invalid input format: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def get_shap_explanations(input_features_scaled, input_features_original, predicted_class_idx=None):
    """Generate SHAP explanations for a prediction - Enhanced for clinical transparency
    
    This function uses SHAP (SHapley Additive exPlanations) to provide transparent,
    interpretable explanations of the AI model's predictions. It helps healthcare
    providers understand which health factors contributed to the risk assessment.
    
    Args:
        input_features_scaled: Scaled input features for the model
        input_features_original: Original (unscaled) input features for display
        predicted_class_idx: Index of the predicted risk class (for multi-class models)
    
    Returns:
        Dictionary containing SHAP explanations with clinical context, or None if unavailable
    """
    global shap_explainer, feature_names, CLINICAL_RANGES
    
    if shap_explainer is None:
        print("⚠️  SHAP explainer not available - explanations will not be generated")
        return None
    
    try:
        # Calculate SHAP values
        shap_values = shap_explainer.shap_values(input_features_scaled)
        
        # Handle multi-class output (SHAP returns array for each class)
        if isinstance(shap_values, list):
            # For multi-class XGBoost, shap_values is a list of arrays (one per class)
            # Use the predicted class if provided, otherwise use mean absolute values
            if predicted_class_idx is not None and 0 <= predicted_class_idx < len(shap_values):
                shap_values = shap_values[predicted_class_idx]
                print(f"✅ Using SHAP values for predicted class index: {predicted_class_idx}")
            else:
                # Use mean absolute values across all classes
                shap_values_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                shap_values = shap_values_combined
                print("✅ Using mean absolute SHAP values across all classes")
            
            # Get first sample if multi-dimensional
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
        else:
            # Single output or array
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Get first sample
        
        # Create explanation dictionary with enhanced information
        explanation = {
            'feature_contributions': {},
            'feature_importance': {},
            'base_value': float(shap_explainer.expected_value) if hasattr(shap_explainer, 'expected_value') else None,
            'explanation_available': True,
            'clinical_context': {}
        }
        
        # Map SHAP values to feature names with original values and clinical context
        for i, feature_name in enumerate(feature_names):
            shap_value = float(shap_values[i])
            original_value = float(input_features_original[0][i])
            
            # Get clinical reference range
            clinical_info = CLINICAL_RANGES.get(feature_name, {})
            normal_range = clinical_info.get('normal', (None, None))
            unit = clinical_info.get('unit', '')
            description = clinical_info.get('description', feature_name)
            
            # Determine if value is within normal range
            within_normal = None
            if normal_range[0] is not None and normal_range[1] is not None:
                within_normal = normal_range[0] <= original_value <= normal_range[1]
            
            # Determine impact level based on absolute SHAP value
            abs_shap = abs(shap_value)
            if abs_shap > 0.5:
                impact = 'high'
            elif abs_shap > 0.2:
                impact = 'medium'
            else:
                impact = 'low'
            
            # Determine clinical significance
            clinical_significance = 'normal'
            if shap_value > 0.3:  # Strong positive contribution to risk
                clinical_significance = 'high_risk_factor'
            elif shap_value > 0.1:
                clinical_significance = 'moderate_risk_factor'
            elif shap_value < -0.3:  # Strong negative contribution (protective)
                clinical_significance = 'protective_factor'
            elif shap_value < -0.1:
                clinical_significance = 'mildly_protective'
            
            explanation['feature_contributions'][feature_name] = {
                'shap_value': round(shap_value, 4),
                'original_value': round(original_value, 2),
                'contribution': 'increases' if shap_value > 0 else 'decreases' if shap_value < 0 else 'neutral',
                'impact': impact,
                'absolute_contribution': round(abs_shap, 4),
                'unit': unit,
                'description': description,
                'normal_range': {'min': normal_range[0], 'max': normal_range[1]} if normal_range[0] is not None else None,
                'within_normal_range': within_normal,
                'clinical_significance': clinical_significance
            }
            
            explanation['feature_importance'][feature_name] = abs_shap
            
            # Store clinical context
            explanation['clinical_context'][feature_name] = {
                'normal_range': f"{normal_range[0]}-{normal_range[1]} {unit}" if normal_range[0] is not None else "N/A",
                'current_value': f"{round(original_value, 2)} {unit}",
                'status': 'Within normal range' if within_normal else 'Outside normal range' if within_normal is False else 'Unknown'
            }
        
        # Sort features by importance (absolute SHAP value)
        sorted_features = sorted(
            explanation['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        explanation['top_contributors'] = [
            {
                'feature': feature,
                'importance': round(importance, 4),
                'contribution': explanation['feature_contributions'][feature]['contribution'],
                'shap_value': explanation['feature_contributions'][feature]['shap_value'],
                'impact': explanation['feature_contributions'][feature]['impact'],
                'original_value': explanation['feature_contributions'][feature]['original_value'],
                'unit': explanation['feature_contributions'][feature]['unit'],
                'clinical_significance': explanation['feature_contributions'][feature]['clinical_significance'],
                'within_normal_range': explanation['feature_contributions'][feature]['within_normal_range']
            }
            for feature, importance in sorted_features
        ]
        
        # Generate clinical summary
        risk_factors = [f for f in explanation['top_contributors'] if f['shap_value'] > 0.1]
        protective_factors = [f for f in explanation['top_contributors'] if f['shap_value'] < -0.1]
        
        explanation['clinical_summary'] = {
            'primary_risk_factors': [
                {
                    'feature': f['feature'],
                    'value': f"{f['original_value']} {f['unit']}",
                    'impact': f['impact'],
                    'within_normal': f['within_normal_range']
                }
                for f in risk_factors[:3]  # Top 3 risk factors
            ],
            'protective_factors': [
                {
                    'feature': f['feature'],
                    'value': f"{f['original_value']} {f['unit']}",
                    'impact': f['impact']
                }
                for f in protective_factors[:2]  # Top 2 protective factors
            ],
            'total_risk_factors': len(risk_factors),
            'total_protective_factors': len(protective_factors)
        }
        
        print(f"✅ SHAP explanations generated successfully for {len(explanation['top_contributors'])} features")
        print(f"   📊 Primary risk factors: {len(risk_factors)}, Protective factors: {len(protective_factors)}")
        return explanation
        
    except Exception as e:
        print(f"⚠️  Error generating SHAP explanations: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_recommendations(risk_level, health_data):
    """Generate personalized recommendations based on predicted risk level and health metrics"""
    risk_level_lower = risk_level.lower().strip()
    
    recommendations = {
        'general': [],
        'immediate_actions': [],
        'monitoring': [],
        'lifestyle': [],
        'medical_care': []
    }
    
    # Extract health metrics for personalized recommendations
    age = health_data.get('Age', 0)
    systolic_bp = health_data.get('SystolicBP', 0)
    diastolic_bp = health_data.get('DiastolicBP', 0)
    blood_sugar = health_data.get('BS', 0)
    body_temp = health_data.get('BodyTemp', 0)
    heart_rate = health_data.get('HeartRate', 0)
    
    if risk_level_lower == 'low risk':
        recommendations['general'] = [
            "Your current health indicators suggest a low-risk pregnancy status.",
            "Continue maintaining your healthy lifestyle and regular prenatal care.",
            "Keep up with scheduled medical appointments and screenings."
        ]
        recommendations['immediate_actions'] = [
            "No immediate medical intervention required.",
            "Continue following your current care plan."
        ]
        recommendations['monitoring'] = [
            "Maintain regular prenatal check-ups (monthly until 28 weeks, then bi-weekly).",
            "Monitor blood pressure, blood sugar, and vital signs as recommended by your healthcare provider.",
            "Track fetal movements daily starting from 28 weeks."
        ]
        recommendations['lifestyle'] = [
            "Maintain a balanced diet rich in folic acid, iron, and calcium.",
            "Engage in moderate exercise (30 minutes most days) as approved by your doctor.",
            "Stay hydrated (8-10 glasses of water daily).",
            "Get adequate sleep (7-9 hours per night).",
            "Avoid alcohol, smoking, and limit caffeine intake."
        ]
        recommendations['medical_care'] = [
            "Continue routine prenatal visits.",
            "Follow standard prenatal vitamin and supplement regimen.",
            "Keep vaccinations up to date as recommended."
        ]
        
    elif risk_level_lower == 'mid risk':
        recommendations['general'] = [
            "Your health indicators show a moderate risk level that requires attention.",
            "Increased monitoring and proactive care are recommended.",
            "Work closely with your healthcare team to manage any concerning factors."
        ]
        recommendations['immediate_actions'] = [
            "Schedule a follow-up appointment with your healthcare provider within 1-2 weeks.",
            "Discuss any concerning symptoms or changes with your doctor immediately."
        ]
        recommendations['monitoring'] = [
            "Increase frequency of prenatal visits (every 2-3 weeks).",
            "Monitor blood pressure daily and keep a log.",
            "Track blood sugar levels if elevated (check 2-4 times daily if recommended).",
            "Monitor fetal movements twice daily.",
            "Watch for signs of preeclampsia: headaches, vision changes, swelling, upper abdominal pain."
        ]
        
        # Personalized recommendations based on specific metrics
        if systolic_bp > 130 or diastolic_bp > 85:
            recommendations['medical_care'].append(
                "Blood pressure monitoring is important. Consider dietary changes (reduce sodium, increase potassium)."
            )
            recommendations['lifestyle'].append(
                "Limit sodium intake to less than 2,300 mg per day. Consider DASH diet principles."
            )
        
        if blood_sugar > 7.8:
            recommendations['medical_care'].append(
                "Blood sugar levels need attention. Discuss glucose monitoring with your healthcare provider."
            )
            recommendations['lifestyle'].append(
                "Follow a balanced meal plan with controlled carbohydrates. Eat smaller, frequent meals."
            )
        
        if heart_rate > 100:
            recommendations['medical_care'].append(
                "Elevated heart rate detected. Discuss with your healthcare provider to rule out any underlying conditions."
            )
            recommendations['lifestyle'].append(
                "Practice stress-reduction techniques: meditation, deep breathing, gentle yoga."
            )
        
        recommendations['lifestyle'].extend([
            "Maintain strict adherence to prenatal care guidelines.",
            "Consider consulting a nutritionist for personalized meal planning.",
            "Limit physical activity intensity; focus on gentle exercises like walking or swimming.",
            "Ensure adequate rest and avoid overexertion."
        ])
        
        recommendations['medical_care'].extend([
            "May require additional ultrasounds or specialized tests.",
            "Consider consultation with maternal-fetal medicine specialist.",
            "Discuss any medication adjustments with your healthcare provider."
        ])
        
    elif risk_level_lower == 'high risk':
        recommendations['general'] = [
            "⚠️ HIGH RISK: Your health indicators require immediate medical attention.",
            "This risk level indicates the need for specialized care and close monitoring.",
            "Do not delay seeking medical care or following your healthcare provider's instructions."
        ]
        recommendations['immediate_actions'] = [
            "🚨 Contact your healthcare provider IMMEDIATELY or go to the emergency department if experiencing severe symptoms.",
            "Do not wait for scheduled appointments - seek care right away if you have concerns.",
            "Have emergency contact numbers readily available."
        ]
        recommendations['monitoring'] = [
            "Requires frequent monitoring (weekly or more frequent visits).",
            "Daily blood pressure monitoring with detailed logs.",
            "Regular blood sugar monitoring (4+ times daily if indicated).",
            "Continuous fetal heart rate monitoring may be necessary.",
            "Watch for emergency warning signs: severe headache, vision changes, chest pain, difficulty breathing, severe abdominal pain, decreased fetal movement."
        ]
        
        # High-risk specific personalized recommendations
        if systolic_bp >= 140 or diastolic_bp >= 90:
            recommendations['immediate_actions'].append(
                "⚠️ Elevated blood pressure detected - this may indicate preeclampsia. Seek immediate medical evaluation."
            )
            recommendations['medical_care'].append(
                "May require blood pressure medication and frequent monitoring. Hospitalization may be necessary."
            )
        
        if blood_sugar >= 11.1:
            recommendations['immediate_actions'].append(
                "⚠️ High blood sugar levels detected - may indicate gestational diabetes requiring immediate management."
            )
            recommendations['medical_care'].append(
                "May require insulin therapy or other diabetes management. Consult endocrinologist if needed."
            )
        
        if body_temp >= 100.4:
            recommendations['immediate_actions'].append(
                "⚠️ Fever detected - contact healthcare provider immediately as this can affect pregnancy."
            )
        
        if heart_rate > 120:
            recommendations['medical_care'].append(
                "Elevated heart rate requires evaluation. May indicate underlying cardiac or other medical conditions."
            )
        
        recommendations['lifestyle'] = [
            "Strict bed rest may be recommended - follow your doctor's specific instructions.",
            "Limit all physical activity unless specifically approved by your healthcare provider.",
            "Follow a medically supervised diet plan.",
            "Avoid all potential stressors and get maximum rest.",
            "Have a support system in place for daily activities."
        ]
        
        recommendations['medical_care'] = [
            "Requires care from a maternal-fetal medicine specialist or high-risk pregnancy team.",
            "May need hospitalization for monitoring and treatment.",
            "Additional diagnostic tests: detailed ultrasounds, non-stress tests, biophysical profiles.",
            "May require medication management for blood pressure, blood sugar, or other conditions.",
            "Consider delivery planning discussions with your healthcare team.",
            "Regular fetal growth monitoring and Doppler studies may be necessary."
        ]
    
    else:
        # Fallback for unknown risk levels
        recommendations['general'] = [
            "Please consult with your healthcare provider for personalized recommendations.",
            "Continue following your current care plan and report any concerns."
        ]
    
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pre-pregnancy')
def pre_pregnancy():
    return render_template('pre.html')

@app.route('/during-pregnancy')
def during_pregnancy():
    return render_template('due.html')

@app.route('/post-pregnancy')
def post_pregnancy():
    return render_template('post.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            'error': 'Prediction model not available.',
            'solution': 'Run XGboost_fixed.py first to train the model'
        }), 503
    
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate patient_id if provided (minimum 3 characters)
    patient_id = data.get('patient_id')
    if patient_id:
        patient_id = str(patient_id).strip()
        if len(patient_id) < 3:
            return jsonify({'error': 'Patient ID must be at least 3 characters long'}), 400
        if len(patient_id) > 50:
            return jsonify({'error': 'Patient ID must be 50 characters or less'}), 400
    
    # Validate input data
    is_valid, validation_result = validate_health_data(data)
    if not is_valid:
        return jsonify({'error': validation_result}), 400
    
    validated_data = validation_result
    
    try:
        # Create features array in correct order (must match training data column order)
        # Training data order: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
        features = np.array([[
            validated_data['Age'],
            validated_data['SystolicBP'],
            validated_data['DiastolicBP'],
            validated_data['BS'],
            validated_data['BodyTemp'],
            validated_data['HeartRate']
        ]])
        
        # Debug: Log input features
        print(f"🔍 Prediction Input - Age: {validated_data['Age']}, SystolicBP: {validated_data['SystolicBP']}, "
              f"DiastolicBP: {validated_data['DiastolicBP']}, BS: {validated_data['BS']}, "
              f"BodyTemp: {validated_data['BodyTemp']}, HeartRate: {validated_data['HeartRate']}")
        
        # Scale features and make prediction
        input_scaled = scaler.transform(features)
        pred_encoded = model.predict(input_scaled)
        pred_label = le.inverse_transform(pred_encoded)
        
        # Debug: Log prediction result
        print(f"📊 Prediction Result - Encoded: {pred_encoded[0]}, Label: {pred_label[0]}")
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = float(np.max(probabilities) * 100)  # Convert to Python float
        
        # Create detailed response with converted types
        risk_level = str(pred_label[0])  # Ensure string type
        
        # Convert probabilities to Python native types
        probabilities_dict = {}
        for i, prob in enumerate(probabilities):
            probabilities_dict[str(le.classes_[i])] = float(prob * 100)  # Convert to Python float
        
        # Get personalized recommendations based on risk level
        recommendations = get_recommendations(risk_level, validated_data)
        
        # Generate SHAP explanations (pass predicted class index for multi-class)
        predicted_class_idx = int(pred_encoded[0])
        shap_explanation = get_shap_explanations(input_scaled, features, predicted_class_idx)
        
        # Don't save to database automatically - user must click Save button
        response = {
            'risk_level': risk_level,
            'confidence': round(confidence, 2),
            'probabilities': probabilities_dict,
            'input_features': validated_data,
            'recommendations': recommendations,
            'saved_to_db': False,
            'shap_explanation': shap_explanation
        }
        
        # Ensure all values are JSON serializable
        response = convert_numpy_types(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'model_loaded': model_loaded,
        'status': 'healthy' if model_loaded else 'unhealthy',
        'message': 'Service is ready' if model_loaded else 'Service unavailable - model not loaded',
        'risk_levels': list(le.classes_) if le else [],
        'shap_available': shap_explainer is not None,
        'explainable_ai': 'SHAP explanations enabled' if shap_explainer is not None else 'SHAP explanations disabled'
    }
    
    # Convert to JSON serializable types
    status = convert_numpy_types(status)
    return jsonify(status)

@app.route('/api/shap/explain', methods=['POST'])
def explain_prediction():
    """Dedicated endpoint for SHAP explanations - allows re-explaining existing predictions
    
    This endpoint provides detailed SHAP explanations for a given set of health parameters.
    Useful for healthcare providers who want to understand model reasoning without making
    a full prediction request.
    """
    if not model_loaded:
        return jsonify({
            'error': 'Prediction model not available.',
            'solution': 'Run XGboost_fixed.py first to train the model'
        }), 503
    
    if shap_explainer is None:
        return jsonify({
            'error': 'SHAP explainer not available',
            'message': 'Explainable AI features are disabled. Please install SHAP library.'
        }), 503
    
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate input data
    is_valid, validation_result = validate_health_data(data)
    if not is_valid:
        return jsonify({'error': validation_result}), 400
    
    validated_data = validation_result
    
    try:
        # Create features array
        features = np.array([[
            validated_data['Age'],
            validated_data['SystolicBP'],
            validated_data['DiastolicBP'],
            validated_data['BS'],
            validated_data['BodyTemp'],
            validated_data['HeartRate']
        ]])
        
        # Scale features
        input_scaled = scaler.transform(features)
        
        # Get prediction for class index
        pred_encoded = model.predict(input_scaled)
        predicted_class_idx = int(pred_encoded[0])
        
        # Generate SHAP explanation
        shap_explanation = get_shap_explanations(input_scaled, features, predicted_class_idx)
        
        if not shap_explanation:
            return jsonify({
                'error': 'Failed to generate SHAP explanation',
                'message': 'SHAP explainer encountered an error'
            }), 500
        
        response = {
            'input_features': validated_data,
            'shap_explanation': shap_explanation,
            'explanation_method': 'SHAP (SHapley Additive exPlanations)',
            'methodology': 'SHAP values explain the output of the machine learning model by showing the marginal contribution of each feature to the prediction. Positive values increase risk, negative values decrease risk.'
        }
        
        # Ensure all values are JSON serializable
        response = convert_numpy_types(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ SHAP explanation error: {str(e)}")
        return jsonify({'error': f'Explanation error: {str(e)}'}), 500

@app.route('/test-prediction', methods=['GET'])
def test_prediction():
    """Test endpoint with sample data"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    sample_data = {
        'Age': 25,
        'SystolicBP': 120,
        'DiastolicBP': 80,
        'BS': 7.0,
        'BodyTemp': 98.0,
        'HeartRate': 75
    }
    
    try:
        features = np.array([[
            sample_data['Age'],
            sample_data['SystolicBP'],
            sample_data['DiastolicBP'],
            sample_data['BS'],
            sample_data['BodyTemp'],
            sample_data['HeartRate']
        ]])
        
        input_scaled = scaler.transform(features)
        pred_encoded = model.predict(input_scaled)
        pred_label = le.inverse_transform(pred_encoded)
        
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = float(np.max(probabilities) * 100)
        
        # Get recommendations for test data
        risk_level = str(pred_label[0])
        recommendations = get_recommendations(risk_level, sample_data)
        
        response = {
            'test_data': sample_data,
            'prediction': risk_level,
            'confidence': round(confidence, 2),
            'status': 'Model is working correctly',
            'recommendations': recommendations
        }
        
        # Convert to JSON serializable types
        response = convert_numpy_types(response)
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Test prediction error: {str(e)}")
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get list of all patients"""
    patients = get_all_patients()
    return jsonify({'patients': patients})

@app.route('/api/patient/<patient_id>/predictions', methods=['GET'])
def get_patient_predictions(patient_id):
    """Get all predictions for a specific patient"""
    month = request.args.get('month', type=int)
    year = request.args.get('year', type=int)
    
    predictions = get_monthly_predictions(patient_id, month, year)
    return jsonify({
        'patient_id': patient_id,
        'month': month,
        'year': year,
        'predictions': predictions,
        'count': len(predictions)
    })

@app.route('/api/patient/<patient_id>/monthly-report', methods=['GET'])
def get_monthly_report(patient_id):
    """Get monthly report for a patient"""
    month = request.args.get('month', type=int)
    year = request.args.get('year', type=int)
    
    if not month or not year:
        return jsonify({'error': 'Month and year parameters are required'}), 400
    
    predictions = get_monthly_predictions(patient_id, month, year)
    
    if not predictions:
        return jsonify({
            'patient_id': patient_id,
            'month': month,
            'year': year,
            'message': 'No predictions found for this month',
            'predictions': []
        })
    
    # Calculate summary statistics
    risk_levels = [p['risk_level'] for p in predictions]
    avg_confidence = sum(p.get('confidence', 0) or 0 for p in predictions) / len(predictions) if predictions else 0
    
    # Get most recent prediction
    latest_prediction = predictions[0] if predictions else None
    
    return jsonify({
        'patient_id': patient_id,
        'month': month,
        'year': year,
        'summary': {
            'total_predictions': len(predictions),
            'average_confidence': round(avg_confidence, 2),
            'risk_levels': {
                'high risk': risk_levels.count('high risk'),
                'mid risk': risk_levels.count('mid risk'),
                'low risk': risk_levels.count('low risk')
            },
            'latest_risk_level': latest_prediction['risk_level'] if latest_prediction else None,
            'latest_confidence': latest_prediction.get('confidence') if latest_prediction else None
        },
        'predictions': predictions
    })

@app.route('/save-prediction', methods=['POST'])
def save_prediction():
    """Save prediction results to database - called manually by user"""
    if not model_loaded:
        return jsonify({
            'error': 'Prediction model not available.',
            'solution': 'Run XGboost_fixed.py first to train the model'
        }), 503
    
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate patient_id (required for saving)
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({'error': 'Patient ID is required to save prediction'}), 400
    
    patient_id = str(patient_id).strip()
    if len(patient_id) < 3:
        return jsonify({'error': 'Patient ID must be at least 3 characters long'}), 400
    if len(patient_id) > 50:
        return jsonify({'error': 'Patient ID must be 50 characters or less'}), 400
    
    # Validate input data
    is_valid, validation_result = validate_health_data(data)
    if not is_valid:
        return jsonify({'error': validation_result}), 400
    
    validated_data = validation_result
    
    # Get risk level and confidence from the request (from previous prediction)
    risk_level = data.get('risk_level')
    confidence = data.get('confidence', 0)
    recommendations = data.get('recommendations', {})
    pregnancy_month = data.get('pregnancy_month')
    
    if not risk_level:
        return jsonify({'error': 'Risk level is required to save prediction'}), 400
    
    # Validate pregnancy_month if provided (should be 1-11)
    if pregnancy_month is not None:
        try:
            pregnancy_month = int(pregnancy_month)
            if pregnancy_month < 1 or pregnancy_month > 11:
                return jsonify({'error': 'Pregnancy month must be between 1 and 11'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Pregnancy month must be a number between 1 and 11'}), 400
    
    # Save to database
    success = save_prediction_to_db(patient_id, validated_data, risk_level, confidence, recommendations, pregnancy_month)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Prediction saved to database successfully',
            'patient_id': patient_id
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to save prediction to database'
        }), 500

@app.route('/api/dashboard/month/<int:month>', methods=['GET'])
def get_dashboard_month(month):
    """Get dashboard data for a specific pregnancy month (1-11)"""
    patient_id = request.args.get('patient_id')
    
    if not patient_id:
        return jsonify({'error': 'patient_id parameter is required'}), 400
    
    # Validate month range (1-11 for pregnancy months)
    if month < 1 or month > 11:
        return jsonify({'error': 'Pregnancy month must be between 1 and 11'}), 400
    
    # Get predictions for this specific pregnancy month
    predictions = get_monthly_predictions(patient_id, pregnancy_month=month)
    
    if not predictions:
        return jsonify({
            'pregnancy_month': month,
            'patient_id': patient_id,
            'has_data': False,
            'data': None,
            'message': f'No predictions found for pregnancy month {month}'
        })
    
    # Get the most recent prediction for this pregnancy month
    latest = predictions[0]  # Most recent prediction for this month
    
    return jsonify({
        'pregnancy_month': month,
        'patient_id': patient_id,
        'has_data': True,
        'data': latest,
        'total_predictions': len(predictions),
        'prediction_date': latest.get('prediction_date') if latest else None
    })

@app.route('/api/dashboard/comparison', methods=['GET'])
def get_monthly_comparison():
    """Get all monthly predictions for a patient for chart comparison with parameters"""
    patient_id = request.args.get('patient_id')
    
    if not patient_id:
        return jsonify({'error': 'patient_id parameter is required'}), 400
    
    # Get all predictions for this patient
    all_predictions = get_monthly_predictions(patient_id)
    
    if not all_predictions:
        return jsonify({
            'patient_id': patient_id,
            'has_data': False,
            'data': [],
            'message': 'No predictions found for this patient'
        })
    
    # Organize predictions by month (1-11)
    monthly_data = {}
    for pred in all_predictions:
        month = pred.get('prediction_month')
        if month is not None:
            # Only keep the most recent prediction for each month
            if month not in monthly_data:
                monthly_data[month] = pred
    
    # Create array for months with data, including all parameters
    chart_data = []
    for month in range(1, 12):
        if month in monthly_data:
            pred = monthly_data[month]
            chart_data.append({
                'month': month,
                'risk_level': pred.get('risk_level', 'N/A'),
                'confidence': pred.get('confidence', 0),
                'prediction_date': str(pred.get('prediction_date', '')),
                'systolic_bp': pred.get('systolic_bp'),
                'diastolic_bp': pred.get('diastolic_bp'),
                'blood_sugar': pred.get('blood_sugar'),
                'body_temp': pred.get('body_temp'),
                'heart_rate': pred.get('heart_rate'),
                'age': pred.get('age'),
                'has_data': True
            })
    
    return jsonify({
        'patient_id': patient_id,
        'has_data': True,
        'data': chart_data,
        'total_months_with_data': len(monthly_data)
    })

if __name__ == '__main__':
    print("🚀 Starting Maternal Health Risk Prediction Server...")
    print("🔧 Loading machine learning models...")
    
    # Load models at startup
    if load_models():
        print("✅ All models loaded successfully!")
    else:
        print("❌ Model loading failed. Running in limited mode.")
        print("💡 Run: python XGboost_fixed.py")
    
    # Initialize database
    print("\n🗄️  Initializing database...")
    if init_database():
        print("✅ Database ready!")
    else:
        print("⚠️  Database initialization failed. App will run without database storage.")
    
    print("\n📊 Available routes:")
    print("   GET  / - Home page")
    print("   GET  /pre-pregnancy - Pre-pregnancy assessment")
    print("   GET  /during-pregnancy - During pregnancy assessment") 
    print("   GET  /post-pregnancy - Post-pregnancy assessment")
    print("   POST /predict - Risk prediction API (includes SHAP explanations)")
    print("   POST /save-prediction - Save prediction to database")
    print("   POST /api/shap/explain - Get SHAP explanations for health parameters")
    print("   GET  /health - Health check (includes SHAP availability status)")
    print("   GET  /test-prediction - Test prediction endpoint")
    print("   GET  /api/patients - Get all patients")
    print("   GET  /api/patient/<id>/predictions - Get patient predictions")
    print("   GET  /api/patient/<id>/monthly-report?month=X&year=Y - Monthly report")
    print("   GET  /api/dashboard/month/<1-11>?patient_id=X - Dashboard month data")
    print("\n🤖 Explainable AI (SHAP):")
    print(f"   Status: {'✅ Enabled' if shap_explainer is not None else '❌ Disabled'}")
    if shap_explainer is not None:
        print("   SHAP provides transparent explanations showing which health factors")
        print("   contribute to each risk prediction, helping healthcare providers")
        print("   understand the AI model's decision-making process.")
    print("\n🌐 Server will start on: http://127.0.0.1:5002")
    
    # Use port 5002 to avoid conflicts
    app.run(debug=True, port=5002, host='0.0.0.0')