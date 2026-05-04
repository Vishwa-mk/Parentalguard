# Database Setup Guide

## Prerequisites
- MySQL Server installed and running
- Python packages: `mysql-connector-python`, `python-dotenv`

## Setup Steps

### 1. Install Required Packages
```bash
pip install mysql-connector-python python-dotenv
```

### 2. Configure Database Connection

Create a `.env` file in the project root with the following:

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=prenatalguard
DB_USER=root
DB_PASSWORD=your_password_here
```

Replace `your_password_here` with your MySQL root password (or leave empty if no password).

### 3. Initialize Database

Run the initialization script:

```bash
python init_database.py
```

This will:
- Create the `prenatalguard` database
- Create `patients` table
- Create `predictions` table with proper indexes

### 4. Start the Application

```bash
python app.py
```

The app will automatically connect to the database on startup.

## Database Schema

### Patients Table
- `patient_id` (VARCHAR, PRIMARY KEY) - Patient identification number
- `name` (VARCHAR) - Patient name
- `age` (INT) - Patient age
- `created_at` (TIMESTAMP) - Record creation time
- `updated_at` (TIMESTAMP) - Last update time

### Predictions Table
- `prediction_id` (INT, AUTO_INCREMENT, PRIMARY KEY)
- `patient_id` (VARCHAR, FOREIGN KEY) - References patients table
- `prediction_date` (DATE) - Date of prediction
- `prediction_month` (INT) - Pregnancy month number (1-11)
- `prediction_year` (INT) - Year
- `age`, `systolic_bp`, `diastolic_bp`, `blood_sugar`, `body_temp`, `heart_rate` (FLOAT)
- `risk_level` (VARCHAR) - Predicted risk level
- `confidence` (FLOAT) - Prediction confidence percentage
- `recommendations` (JSON) - Stored recommendations
- `created_at` (TIMESTAMP) - Record creation time

## API Endpoints

### Save Prediction
When making a prediction, include `patient_id` in the request:
```json
{
  "patient_id": "P001",
  "name": "Jane Doe",
  "Age": 28,
  "SystolicBP": 120,
  "DiastolicBP": 80,
  "BS": 7.0,
  "BodyTemp": 98.0,
  "HeartRate": 75
}
```

### Get Monthly Report
```
GET /api/patient/<patient_id>/monthly-report?month=1&year=2024
```

### Get Dashboard Data
```
GET /api/dashboard/month/<1-11>?patient_id=P001
```

## Troubleshooting

### Connection Errors
- Verify MySQL server is running: `mysql -u root -p`
- Check `.env` file has correct credentials
- Ensure MySQL allows connections from localhost

### Table Creation Errors
- Make sure you have CREATE DATABASE privileges
- Check if database already exists and drop it if needed

### No Data in Dashboard
- Ensure predictions are saved with a `patient_id`
- Check that the patient_id matches when fetching data
- Verify data exists in the database using MySQL client

