"""
Setup script for AgriSense
"""
import os

def setup_environment():
    """Setup the environment"""
    print("Setting up AgriSense...")

    # Create necessary directories
    directories = [
        'models/1',
        'models/2',
        'models/3',
        'models/4',
        'templates',
        'static/uploads',
        'static/history',
        'static/css',
        'static/js'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_ENV=development
FLASK_DEBUG=1

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=AgriSense

# Google Maps API
GOOGLE_MAPS_API_KEY=AIzaSyDrSy2r_gHTnLbILR1joImzwErOp56G-7M

# Application Settings
UPLOAD_FOLDER=static/uploads
MAX_CONTENT_LENGTH=16777216  # 16MB
""")
        print("Created: .env")

    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run database setup: python db-setup.py")
    print("3. Start the application: python app.py")
    print("4. Access at: http://localhost:5000")


if __name__ == '__main__':
    setup_environment()