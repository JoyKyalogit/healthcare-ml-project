-- Create cleaned table if not present
CREATE TABLE IF NOT EXISTS patients_cleaned (
    id SERIAL PRIMARY KEY,
    age INTEGER NOT NULL,
    gender VARCHAR(50) NOT NULL,
    blood_type VARCHAR(10) NOT NULL,
    medical_condition VARCHAR(100) NOT NULL,
    date_of_admission DATE NOT NULL,
    insurance_provider VARCHAR(100) NOT NULL,
    billing_amount NUMERIC(12, 2) NOT NULL,
    admission_type VARCHAR(50) NOT NULL,
    medication VARCHAR(100) NOT NULL,
    length_of_stay INTEGER NOT NULL,
    test_results VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    accuracy FLOAT NOT NULL,
    precision FLOAT NOT NULL,
    recall FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Useful checks
SELECT COUNT(*) AS cleaned_records FROM patients_cleaned;
SELECT model_name, accuracy, precision, recall, f1_score, trained_at
FROM model_metrics
ORDER BY trained_at DESC;
