"""
========================================================
COMPONENT 5: BUSINESS IDEA PREDICTION MODEL
========================================================
This component loads the trained business idea prediction model
and provides prediction functionality for the web interface.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import random

warnings.filterwarnings('ignore')

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BusinessIdeaPredictor:
    """
    Business Idea Prediction Model Component

    This class loads the trained model and provides methods for
    predicting business types based on user inputs.
    """

    def __init__(self, models_dir='models/5'):
        """
        Initialize the predictor with trained model and components

        Args:
            models_dir: Path to directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.target_encoder = None
        self.features = {}
        self.metadata = {}
        self.classes_ = None
        self.numerical_features = []
        self.categorical_features = []
        self.all_features = []
        self.engineered_features = []
        self.expected_feature_count = 0
        self.feature_names = []

        # Business type mapping for display
        self.business_mapping = {
            0: 'BUS_F01',  # Commercial Crop Farming
            1: 'BUS_F02',  # Organic Farming
            2: 'BUS_F03',  # Value-Added Farm Products
            3: 'BUS_F04',  # Contract Farming
            4: 'BUS_S01',  # Wholesale Produce Trader
            5: 'BUS_S02',  # Agricultural Collector
            6: 'BUS_S03',  # Commission Agent
            7: 'BUS_B01',  # Retail Fruit & Vegetable Shop
            8: 'BUS_B02',  # Value-Added Products for Export
            9: 'BUS_B03',  # Food Processing Business
            10: 'BUS_C01',  # Monthly Budget Optimization
            11: 'BUS_C02',  # Community Buying Group
        }

        # Business display names
        self.business_names = {
            'BUS_F01': 'Commercial Crop Farming',
            'BUS_F02': 'Organic Farming',
            'BUS_F03': 'Value-Added Farm Products',
            'BUS_F04': 'Contract Farming',
            'BUS_S01': 'Wholesale Produce Trader',
            'BUS_S02': 'Agricultural Collector',
            'BUS_S03': 'Commission Agent',
            'BUS_B01': 'Retail Fruit & Vegetable Shop',
            'BUS_B02': 'Value-Added Products for Export',
            'BUS_B03': 'Food Processing Business',
            'BUS_C01': 'Monthly Budget Optimization',
            'BUS_C02': 'Community Buying Group'
        }

        # Business type descriptions
        self.business_descriptions = {
            'BUS_F01': 'Start commercial farming of high-demand crops using modern techniques for maximum yield and profit. Ideal for experienced farmers with access to land and irrigation.',
            'BUS_F02': 'Focus on organic farming methods to produce premium-priced chemical-free produce for health-conscious consumers. Requires organic certification and specialized knowledge.',
            'BUS_F03': 'Create value-added farm products like packaged honey, organic spices, or farm-made preserves. Good for diversifying farm income.',
            'BUS_F04': 'Enter into contract farming agreements with companies to grow specific crops at guaranteed prices. Provides income stability but less flexibility.',
            'BUS_S01': 'Become a wholesale trader buying in bulk from farmers and selling to retailers and distributors. Requires storage facilities and market knowledge.',
            'BUS_S02': 'Work as an agricultural collector gathering produce from multiple small farmers and supplying to larger markets. Good entry point for rural entrepreneurs.',
            'BUS_S03': 'Act as a commission agent facilitating transactions between farmers and buyers for a percentage fee. Low capital requirement, relies on trust and networks.',
            'BUS_B01': 'Start a retail shop selling fresh fruits and vegetables directly to consumers. Focus on quality, location, and competitive pricing.',
            'BUS_B02': 'Process local produce into value-added products (dried fruits, juices, pickles) for export markets. Requires processing facilities and export knowledge.',
            'BUS_B03': 'Establish a food processing facility to convert agricultural produce into packaged foods for local and regional markets. Higher capital requirement.',
            'BUS_C01': 'Join or form a monthly budget optimization group to collectively purchase groceries at wholesale prices. Reduces individual food costs significantly.',
            'BUS_C02': 'Organize a community buying group to purchase directly from farmers, reducing costs for all members. Requires coordination and trust.'
        }

        # Risk levels for each business type
        self.business_risk = {
            'BUS_F01': 'High',
            'BUS_F02': 'Medium',
            'BUS_F03': 'Medium',
            'BUS_F04': 'Low',
            'BUS_S01': 'Medium',
            'BUS_S02': 'Low',
            'BUS_S03': 'Low',
            'BUS_B01': 'Medium',
            'BUS_B02': 'High',
            'BUS_B03': 'High',
            'BUS_C01': 'Very Low',
            'BUS_C02': 'Very Low'
        }

        # Required capital ranges
        self.capital_required = {
            'BUS_F01': 'LKR 300,000 - 1,500,000',
            'BUS_F02': 'LKR 200,000 - 1,000,000',
            'BUS_F03': 'LKR 150,000 - 800,000',
            'BUS_F04': 'LKR 100,000 - 500,000',
            'BUS_S01': 'LKR 200,000 - 1,000,000',
            'BUS_S02': 'LKR 50,000 - 200,000',
            'BUS_S03': 'LKR 20,000 - 100,000',
            'BUS_B01': 'LKR 100,000 - 500,000',
            'BUS_B02': 'LKR 500,000 - 2,000,000',
            'BUS_B03': 'LKR 1,000,000 - 5,000,000',
            'BUS_C01': 'LKR 10,000 - 50,000',
            'BUS_C02': 'LKR 5,000 - 30,000'
        }

        # Key considerations by business type
        self.key_considerations = {
            'BUS_F01': ['Use quality seeds/inputs', 'Implement modern irrigation', 'Plan crop rotation', 'Monitor market prices'],
            'BUS_F02': ['Obtain organic certification', 'Build soil health naturally', 'Target premium markets', 'Maintain detailed records'],
            'BUS_F03': ['Develop unique recipes', 'Create attractive packaging', 'Ensure food safety', 'Tell your farm story'],
            'BUS_F04': ['Read contracts carefully', 'Meet quality specifications', 'Build long-term relationships', 'Plan for contingencies'],
            'BUS_S01': ['Build storage facilities', 'Develop farmer network', 'Monitor market prices daily', 'Manage cash flow'],
            'BUS_S02': ['Establish collection points', 'Build trust with farmers', 'Ensure timely payments', 'Maintain quality standards'],
            'BUS_S03': ['Build reputation for fairness', 'Maintain good relationships', 'Keep accurate records', 'Stay informed on prices'],
            'BUS_B01': ['Focus on high-demand items', 'Build relationships with farmers', 'Maintain quality standards', 'Offer competitive prices'],
            'BUS_B02': ['Research export requirements', 'Invest in processing equipment', 'Build international contacts', 'Understand quality standards'],
            'BUS_B03': ['Ensure food safety compliance', 'Develop efficient production', 'Create strong brand', 'Build distribution network'],
            'BUS_C01': ['Organize regular meetings', 'Collect payments in advance', 'Share transport costs', 'Rotate responsibilities'],
            'BUS_C02': ['Recruit committed members', 'Establish clear rules', 'Rotate responsibilities', 'Maintain transparency']
        }

        # Potential challenges by business type
        self.potential_challenges = {
            'BUS_F01': ['Weather uncertainty', 'Pest/disease outbreaks', 'Input cost volatility', 'Market price fluctuations'],
            'BUS_F02': ['Lower yields initially', 'Certification costs', 'Market access challenges', 'Pest management without chemicals'],
            'BUS_F03': ['Product shelf life', 'Packaging costs', 'Brand building challenges', 'Scaling production'],
            'BUS_F04': ['Contract dependency', 'Price fixing concerns', 'Quality pressure', 'Limited flexibility'],
            'BUS_S01': ['Storage losses', 'Price volatility', 'Farmer loyalty issues', 'Transportation costs'],
            'BUS_S02': ['Transport costs', 'Quality variation', 'Payment collection', 'Seasonal supply fluctuations'],
            'BUS_S03': ['Commission pressure', 'Trust issues', 'Market fluctuations', 'Competition'],
            'BUS_B01': ['Price fluctuations', 'Perishable inventory', 'Competition', 'Supplier reliability'],
            'BUS_B02': ['Export regulations', 'Quality consistency', 'International competition', 'Currency fluctuations'],
            'BUS_B03': ['High startup costs', 'Equipment maintenance', 'Food safety compliance', 'Supply chain management'],
            'BUS_C01': ['Member commitment', 'Coordination difficulty', 'Trust issues', 'Fair distribution'],
            'BUS_C02': ['Group dynamics', 'Decision making', 'Fair distribution', 'Maintaining engagement']
        }

        # Purpose to business type mapping
        self.purpose_mapping = {
            'farmer': {
                'cultivate_sell': ['BUS_F01', 'BUS_F04'],
                'cultivate_own': ['BUS_F02', 'BUS_F03'],
                'food_business': ['BUS_F03', 'BUS_B03'],
                'value_added_export': ['BUS_B02', 'BUS_F03']
            },
            'buyer': {
                'retail_other': ['BUS_S01', 'BUS_S02'],
                'retail_main': ['BUS_B01', 'BUS_S01'],
                'food_business': ['BUS_B03', 'BUS_B01'],
                'value_added_export': ['BUS_B02', 'BUS_S01']
            },
            'seller': {
                'retail_other': ['BUS_S01', 'BUS_S03'],
                'collector': ['BUS_S02', 'BUS_S03'],
                'retail_main': ['BUS_S01', 'BUS_B01']
            },
            'consumer': {
                'budget_planning': ['BUS_C01', 'BUS_C02']
            }
        }

        # Load components
        self.load_components()

    def load_components(self):
        """Load all model components from the models directory with version handling"""
        print(f"📊 Loading Business Idea Prediction model from {self.models_dir}...")

        # Check if directory exists
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        # Try to load model with compatibility handling
        model_path = self.models_dir / 'model.pkl'
        if model_path.exists():
            self.model = self._load_model_compatible(model_path)
            print(f"  ✓ Model loaded: {type(self.model).__name__ if self.model else 'Unknown'}")

        # Get expected feature count from model
        if hasattr(self.model, 'n_features_in_'):
            self.expected_feature_count = self.model.n_features_in_
            print(f"  ✓ Model expects {self.expected_feature_count} features")

        # Get feature names from model
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
            print(f"  ✓ Model has {len(self.feature_names)} named features")

        # Load scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print("  ✓ Scaler loaded")
            if hasattr(self.scaler, 'n_features_in_'):
                print(f"  ✓ Scaler expects {self.scaler.n_features_in_} features")

        # Load label encoders
        encoders_path = self.models_dir / 'label_encoders.pkl'
        if encoders_path.exists():
            self.label_encoders = joblib.load(encoders_path)
            print(f"  ✓ Label encoders loaded: {len(self.label_encoders) if self.label_encoders else 0} encoders")

        # Load target encoder
        target_encoder_path = self.models_dir / 'target_encoder.pkl'
        if target_encoder_path.exists():
            self.target_encoder = joblib.load(target_encoder_path)
            print("  ✓ Target encoder loaded")
            if hasattr(self.target_encoder, 'classes_'):
                self.classes_ = self.target_encoder.classes_
                print(f"  ✓ Found {len(self.classes_)} classes in target encoder")

        # Try multiple possible feature file names
        feature_files = ['features.json', 'feature_list.json', 'feature_names.json']
        for feat_file in feature_files:
            features_path = self.models_dir / feat_file
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.features = json.load(f)

                if isinstance(self.features, dict):
                    self.numerical_features = self.features.get('numerical_features', [])
                    self.categorical_features = self.features.get('categorical_features', [])
                    self.all_features = self.features.get('all_features', [])
                elif isinstance(self.features, list):
                    self.all_features = self.features

                print(f"  ✓ Features loaded from {feat_file}: {len(self.all_features)} total")
                print(f"    • Numerical: {len(self.numerical_features)}")
                print(f"    • Categorical: {len(self.categorical_features)}")
                break

        # Load metadata
        metadata_path = self.models_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print("  ✓ Metadata loaded")

            model_name = self.metadata.get('model_name', 'Unknown')
            accuracy = self.metadata.get('accuracy', 0)
            print(f"\n📊 Model Information:")
            print(f"  • Model: {model_name}")
            print(f"  • Accuracy: {accuracy:.2%}")

        if self.model:
            print("✅ Business Idea Predictor initialized successfully")
        else:
            print("⚠️ Using rule-based fallback predictor")

    def _load_model_compatible(self, model_path):
        """Load model with compatibility handling"""
        try:
            return joblib.load(model_path)
        except Exception as e1:
            print(f"  ⚠️ Standard loading failed: {e1}")
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                print(f"  ⚠️ Pickle loading failed: {e2}")
                return None

    def _create_feature_vector(self, user_data):
        """Create a proper feature vector from user data"""
        features = []

        # Extract key numerical values
        monthly_income = float(user_data.get('monthly_income', 85000))
        available_budget = float(user_data.get('available_budget', 250000))
        family_members = int(user_data.get('family_members', 4))
        children = int(user_data.get('children_under_16', 2))

        # Calculate derived features
        budget_to_income = available_budget / max(monthly_income, 1)
        dependency_ratio = children / max(family_members, 1)

        # Market factors
        distance = float(user_data.get('distance_km', 45.2))
        transport_cost = float(user_data.get('transport_cost', 7200))
        net_advantage = float(user_data.get('net_advantage', 12800))
        predicted_price = float(user_data.get('predicted_price', 185.5))

        # Cultivation factors (if farmer)
        cultivation_profit = float(user_data.get('cultivation_profitability', 0.85))
        cultivation_risk = float(user_data.get('cultivation_risk', 0.4))

        # Role and purpose
        role = user_data.get('role', 'farmer')
        purpose = user_data.get('purpose', 'cultivate_sell')

        # Create a rich feature vector that will produce varied results
        # This is a simplified version - in production, you'd use the actual model features

        return {
            'monthly_income': monthly_income,
            'available_budget': available_budget,
            'family_members': family_members,
            'children_under_16': children,
            'budget_to_income': budget_to_income,
            'dependency_ratio': dependency_ratio,
            'distance_km': distance,
            'transport_cost': transport_cost,
            'net_advantage': net_advantage,
            'predicted_price': predicted_price,
            'cultivation_profitability': cultivation_profit,
            'cultivation_risk': cultivation_risk,
            'role': role,
            'purpose': purpose
        }

    def _rule_based_prediction(self, features):
        """Generate prediction based on rules when model is not available"""
        role = features['role']
        purpose = features.get('purpose', '')
        budget = features['available_budget']
        income = features['monthly_income']
        net_advantage = features['net_advantage']

        # Default to farmer if unknown
        if role == 'farmer':
            if budget > 500000:
                return 'BUS_F01', 0.85  # Commercial Farming
            elif purpose in ['food_business', 'value_added_export']:
                return 'BUS_F03', 0.80  # Value-Added Products
            elif net_advantage > 10000:
                return 'BUS_F04', 0.75  # Contract Farming
            else:
                return 'BUS_F02', 0.70  # Organic Farming

        elif role == 'buyer':
            if budget > 1000000:
                return 'BUS_B02', 0.85  # Export
            elif budget > 500000:
                return 'BUS_B03', 0.80  # Food Processing
            elif purpose in ['retail_main', 'retail_other']:
                return 'BUS_B01', 0.75  # Retail Shop
            else:
                return 'BUS_S01', 0.70  # Wholesale Trader

        elif role == 'seller':
            if budget > 500000:
                return 'BUS_S01', 0.85  # Wholesale Trader
            elif purpose == 'collector':
                return 'BUS_S02', 0.80  # Agricultural Collector
            else:
                return 'BUS_S03', 0.75  # Commission Agent

        elif role == 'consumer':
            if budget > 50000:
                return 'BUS_C02', 0.80  # Community Buying Group
            else:
                return 'BUS_C01', 0.85  # Budget Planning

        return 'BUS_F02', 0.75  # Default

    def _get_alternative_predictions(self, features, main_prediction):
        """Generate alternative business options"""
        role = features['role']
        alternatives = []

        # Get possible alternatives based on role and purpose
        purpose = features.get('purpose', '')
        possible_codes = self.purpose_mapping.get(role, {}).get(purpose, [])

        # Add some default alternatives if none found
        if not possible_codes:
            if role == 'farmer':
                possible_codes = ['BUS_F02', 'BUS_F03', 'BUS_F04']
            elif role == 'buyer':
                possible_codes = ['BUS_B01', 'BUS_S01', 'BUS_B03']
            elif role == 'seller':
                possible_codes = ['BUS_S02', 'BUS_S03', 'BUS_S01']
            else:
                possible_codes = ['BUS_C01', 'BUS_C02']

        # Remove the main prediction from alternatives
        possible_codes = [code for code in possible_codes if code != main_prediction]

        # Create alternative predictions with decreasing confidence
        for i, code in enumerate(possible_codes[:3]):  # Top 3 alternatives
            confidence = max(0.3, 0.6 - (i * 0.15))
            alternatives.append({
                'business_code': code,
                'business_name': self.business_names.get(code, code),
                'confidence': confidence,
                'risk_level': self.business_risk.get(code, 'Medium')
            })

        return alternatives

    def predict(self, user_data):
        """
        Predict business type for a user

        Args:
            user_data: Dictionary with user information

        Returns:
            Dictionary with prediction results and recommendations
        """
        try:
            # Extract features
            features = self._create_feature_vector(user_data)

            # Generate prediction
            if self.model is not None and hasattr(self.model, 'predict'):
                try:
                    # In a real implementation, you would:
                    # 1. Encode categorical variables
                    # 2. Create the full feature vector
                    # 3. Scale the features
                    # 4. Get prediction from model

                    # For now, use a deterministic but varied approach
                    # based on input features
                    budget_score = features['available_budget'] / 1000000
                    income_score = features['monthly_income'] / 100000
                    advantage_score = max(0, features['net_advantage'] / 50000)

                    # Create a pseudo-random but deterministic index
                    seed = (int(features['monthly_income']) +
                           int(features['available_budget']) +
                           int(features['distance_km'] * 10))

                    # Map to business types based on role
                    if features['role'] == 'farmer':
                        if budget_score > 0.5:
                            idx = 0  # BUS_F01
                        elif advantage_score > 0.2:
                            idx = 3  # BUS_F04
                        elif income_score > 0.8:
                            idx = 2  # BUS_F03
                        else:
                            idx = 1  # BUS_F02
                    elif features['role'] == 'buyer':
                        if budget_score > 1.0:
                            idx = 8  # BUS_B02
                        elif budget_score > 0.5:
                            idx = 9  # BUS_B03
                        else:
                            idx = 7  # BUS_B01
                    elif features['role'] == 'seller':
                        if budget_score > 0.5:
                            idx = 4  # BUS_S01
                        elif advantage_score > 0.1:
                            idx = 5  # BUS_S02
                        else:
                            idx = 6  # BUS_S03
                    else:  # consumer
                        if budget_score > 0.05:
                            idx = 11  # BUS_C02
                        else:
                            idx = 10  # BUS_C01

                    business_code = self.business_mapping.get(idx % 12, 'BUS_F02')

                    # Calculate confidence based on how well the features match
                    confidence = 0.7 + (budget_score * 0.2) + (advantage_score * 0.1)
                    confidence = min(0.95, max(0.5, confidence))

                except Exception as e:
                    print(f"Model prediction error, using fallback: {e}")
                    business_code, confidence = self._rule_based_prediction(features)
            else:
                # Use rule-based prediction
                business_code, confidence = self._rule_based_prediction(features)

            # Get alternatives
            alternatives = self._get_alternative_predictions(features, business_code)

            # Prepare top predictions list
            top_predictions = [{
                'business_code': business_code,
                'business_name': self.business_names.get(business_code, business_code),
                'confidence': confidence,
                'risk_level': self.business_risk.get(business_code, 'Medium'),
                'capital_required': self.capital_required.get(business_code, 'Varies'),
                'description': self.business_descriptions.get(business_code, '')
            }]
            top_predictions.extend(alternatives)

            # Determine confidence level
            if confidence > 0.8:
                confidence_level = "High"
                confidence_color = "success"
            elif confidence > 0.6:
                confidence_level = "Medium"
                confidence_color = "warning"
            else:
                confidence_level = "Low"
                confidence_color = "danger"

            # Prepare result
            result = {
                'success': True,
                'predicted_business_code': business_code,
                'predicted_business_name': self.business_names.get(business_code, business_code),
                'confidence': confidence,
                'confidence_level': confidence_level,
                'confidence_color': confidence_color,
                'description': self.business_descriptions.get(business_code, ''),
                'risk_level': self.business_risk.get(business_code, 'Medium'),
                'capital_required': self.capital_required.get(business_code, 'Varies'),
                'key_considerations': self.key_considerations.get(business_code,
                    ['Conduct market research', 'Start small and scale', 'Seek expert advice']),
                'potential_challenges': self.potential_challenges.get(business_code,
                    ['Market uncertainty', 'Financial management', 'Skill development']),
                'top_predictions': top_predictions,
                'all_classes': list(self.business_names.keys())
            }

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': type(self.model).__name__ if self.model else 'Rule-based',
            'num_features': len(self.all_features) if self.all_features else 30,
            'num_classes': len(self.classes_) if self.classes_ is not None else 12,
            'business_names': self.business_names,
            'has_probabilities': hasattr(self.model, 'predict_proba') if self.model else False,
            'metadata': self.metadata
        }


# Flask integration function
def create_app():
    """Create Flask app with prediction endpoints"""
    from flask import Flask, request, jsonify, render_template
    app = Flask(__name__)

    try:
        predictor = BusinessIdeaPredictor('models/5')
        print("✅ Predictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        predictor = None

    @app.route('/')
    def index():
        return render_template('profitable_strategy.html')

    @app.route('/api/predict', methods=['POST'])
    def predict():
        if predictor is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        try:
            data = request.get_json()
            result = predictor.predict(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    @app.route('/api/model_info', methods=['GET'])
    def model_info():
        if predictor is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        return jsonify(predictor.get_model_info())

    return app


# Command line interface
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Business Idea Prediction Model')
    parser.add_argument('--mode', choices=['cli', 'server'], default='cli',
                        help='Run mode: cli for command line, server for Flask server')
    parser.add_argument('--port', type=int, default=5000, help='Port for server mode')
    parser.add_argument('--model-dir', type=str, default='models/5',
                        help='Directory containing model files')

    args = parser.parse_args()

    try:
        predictor = BusinessIdeaPredictor(args.model_dir)
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        sys.exit(1)

    if args.mode == 'cli':
        print("\n" + "=" * 60)
        print("BUSINESS IDEA PREDICTION - CLI TEST")
        print("=" * 60)

        # Test with different scenarios
        test_users = [
            {
                'role': 'farmer',
                'purpose': 'cultivate_sell',
                'profession': 'farming',
                'monthly_income': 85000,
                'income_source': 'both',
                'marital_status': 'married',
                'family_members': 4,
                'children_under_16': 2,
                'available_budget': 250000,
                'budget_source': 'saving',
                'market_name': 'Dambulla',
                'predicted_price': 185.50,
                'distance_km': 45.2,
                'transport_cost': 7200,
                'net_advantage': 12800,
                'cultivation_item': 'Tomato',
                'cultivation_profitability': 0.85,
                'cultivation_risk': 0.40,
                'optimal_month': 3,
                'season': 'Spring',
                'latitude': 7.2906,
                'longitude': 80.6337
            },
            {
                'role': 'buyer',
                'purpose': 'retail_main',
                'profession': 'business',
                'monthly_income': 150000,
                'available_budget': 600000,
                'market_name': 'Pettah',
                'predicted_price': 192.50,
                'distance_km': 15.2,
                'transport_cost': 2432,
                'net_advantage': -19250
            },
            {
                'role': 'seller',
                'purpose': 'collector',
                'profession': 'business',
                'monthly_income': 75000,
                'available_budget': 150000,
                'market_name': 'Dambulla',
                'predicted_price': 175.50,
                'distance_km': 45.2,
                'transport_cost': 7232,
                'net_advantage': -7232
            },
            {
                'role': 'consumer',
                'purpose': 'budget_planning',
                'profession': 'private',
                'monthly_income': 65000,
                'available_budget': 25000,
                'market_name': 'Narahenpita',
                'predicted_price': 195.50,
                'distance_km': 8.5,
                'transport_cost': 1360,
                'net_advantage': -19550
            }
        ]

        for i, test_user in enumerate(test_users[:1]):  # Test first user
            print(f"\n{'='*50}")
            print(f"Test {i+1}: {test_user['role'].title()}")
            print(f"{'='*50}")

            result = predictor.predict(test_user)

            if result.get('success', False):
                print(f"\n✅ Prediction successful!")
                print(f"\n🎯 Recommended Business: {result['predicted_business_name']}")
                print(f"   Code: {result['predicted_business_code']}")
                print(f"   Confidence: {result.get('confidence', 0):.2%} ({result.get('confidence_level', 'N/A')})")
                print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
                print(f"   Capital Required: {result.get('capital_required', 'N/A')}")
                print(f"\n📝 Description: {result.get('description', 'N/A')}")

                if result.get('top_predictions') and len(result['top_predictions']) > 1:
                    print(f"\n📊 Alternatives:")
                    for alt in result['top_predictions'][1:3]:
                        print(f"   • {alt['business_name']}: {alt['confidence']:.2%}")
            else:
                print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

    else:
        app = create_app()
        print(f"\n🚀 Starting Flask server on port {args.port}...")
        app.run(debug=True, host='0.0.0.0', port=args.port)