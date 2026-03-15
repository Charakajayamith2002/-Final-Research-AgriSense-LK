"""
Database Setup Script for MongoDB
"""
from datetime import datetime

from bson import ObjectId

from app import logger
from db_config import MongoDBHandler


# Add these methods to your MongoDBHandler class

def create_user(self, user_data):
    """Create a new user in the database"""
    try:
        result = self.users_collection.insert_one(user_data)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        return None


def get_user_by_email(self, email):
    """Get user by email"""
    try:
        return self.users_collection.find_one({'email': email})
    except Exception as e:
        logger.error(f"Error getting user by email: {str(e)}")
        return None


def get_user_by_username(self, username):
    """Get user by username"""
    try:
        return self.users_collection.find_one({'username': username})
    except Exception as e:
        logger.error(f"Error getting user by username: {str(e)}")
        return None


def get_user_by_id(self, user_id):
    """Get user by ID"""
    try:
        return self.users_collection.find_one({'_id': ObjectId(user_id)})
    except Exception as e:
        logger.error(f"Error getting user by ID: {str(e)}")
        return None


def update_user_login(self, user_id):
    """Update user's last login timestamp"""
    try:
        self.users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'last_login': datetime.now()}}
        )
        return True
    except Exception as e:
        logger.error(f"Error updating user login: {str(e)}")
        return False


def update_user_profile(self, user_id, username=None, user_type=None):
    """Update user profile information"""
    try:
        update_data = {}
        if username:
            update_data['username'] = username
        if user_type:
            update_data['user_type'] = user_type

        if update_data:
            self.users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': update_data}
            )
        return True
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        return False


def save_prediction(self, prediction_data):
    """Save prediction to history and link to user"""
    try:
        # Save to predictions collection
        result = self.predictions_collection.insert_one(prediction_data)

        # Also add reference to user's history
        if 'user_id' in prediction_data:
            self.users_collection.update_one(
                {'_id': ObjectId(prediction_data['user_id'])},
                {'$push': {'history': str(result.inserted_id)}}
            )

        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")
        return None


def get_user_history(self, user_id):
    """Get prediction history for a user"""
    try:
        # Get all predictions for this user
        predictions = list(self.predictions_collection.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(50))

        # Convert ObjectId to string for JSON serialization
        for pred in predictions:
            pred['_id'] = str(pred['_id'])

        return predictions
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        return []

def setup_database():
    """Initialize database with collections and sample data"""

    print("=" * 60)
    print("AGRISENSE DATABASE SETUP")
    print("=" * 60)

    # Initialize MongoDB handler
    db_handler = MongoDBHandler('AgriSense')

    # Create collections
    print("\nCreating collections...")
    db_handler.initialize_database()

    # Insert sample market data
    print("\nInserting sample market data...")
    sample_markets = [
        {
            'market': 'Pettah',
            'item': 'Beans',
            'price_type': 'Retail',
            'price': 250.0,
            'date': '2024-01-15',
            'category': 'Vegetables',
            'origin_type': 'Local'
        },
        {
            'market': 'Dambulla',
            'item': 'Tomato',
            'price_type': 'Wholesale',
            'price': 150.0,
            'date': '2024-01-15',
            'category': 'Vegetables',
            'origin_type': 'Local'
        },
        {
            'market': 'Narahenpita',
            'item': 'Carrot',
            'price_type': 'Retail',
            'price': 220.0,
            'date': '2024-01-15',
            'category': 'Vegetables',
            'origin_type': 'Local'
        }
    ]

    for market_data in sample_markets:
        db_handler.save_market_data(market_data)

    # Insert sample user
    print("\nCreating sample user...")
    sample_user = {
        'username': 'demo_user',
        'email': 'demo@agrisense.com',
        'created_at': '2024-01-01',
        'user_type': 'farmer',
        'location': 'Kandy District'
    }

    db_handler.db.users.insert_one(sample_user)

    # Insert sample cultivation history
    print("\nCreating sample cultivation history...")
    sample_cultivation = {
        'user_id': 'demo_user',
        'crop': 'Tomato',
        'planting_date': '2023-12-01',
        'harvest_date': '2024-02-01',
        'yield': 500.0,
        'profit': 25000.0,
        'market': 'Dambulla'
    }

    db_handler.db.cultivation_history.insert_one(sample_cultivation)

    # Get collection statistics
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)

    stats = db_handler.get_collection_stats()
    for collection, data in stats.items():
        print(f"{collection}: {data['count']} documents")

    print("\n" + "=" * 60)
    print("DATABASE SETUP COMPLETE!")
    print("=" * 60)

    # Close connection
    db_handler.close_connection()

if __name__ == '__main__':
    setup_database()