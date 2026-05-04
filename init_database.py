#!/usr/bin/env python3
"""
Database Initialization Script
Run this script to set up the MySQL database for PrenatalGuard
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': int(os.getenv('DB_PORT', 3306))
}

DB_NAME = os.getenv('DB_NAME', 'prenatalguard')

def init_database():
    """Initialize database and create tables"""
    connection = None
    cursor = None
    try:
        # Connect to MySQL server (without database)
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        print(f"✅ Connected to MySQL server at {DB_CONFIG['host']}")
        
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        print(f"✅ Database '{DB_NAME}' created or already exists")
        
        # Use the database
        cursor.execute(f"USE {DB_NAME}")
        
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
        print("✅ Patients table created")
        
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
        print("✅ Predictions table created")
        
        connection.commit()
        print("\n🎉 Database initialization completed successfully!")
        print(f"\nDatabase: {DB_NAME}")
        print("Tables created:")
        print("  - patients")
        print("  - predictions")
        
        return True
        
    except Error as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure MySQL server is installed and running")
        print("   2. Check your .env file has correct credentials")
        print("   3. Try: brew services start mysql (on macOS)")
        print("   4. Or: sudo systemctl start mysql (on Linux)")
        return False
    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            print("\n✅ Database connection closed")

if __name__ == '__main__':
    print("🚀 Initializing PrenatalGuard Database...")
    print(f"Host: {DB_CONFIG['host']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"Database: {DB_NAME}\n")
    
    if not DB_CONFIG['password']:
        print("⚠️  Warning: DB_PASSWORD is empty. Make sure your MySQL root user doesn't require a password.")
    
    init_database()

