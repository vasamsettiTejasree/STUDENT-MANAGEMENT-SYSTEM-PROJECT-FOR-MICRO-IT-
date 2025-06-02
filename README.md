# STUDENT-MANAGEMENT-SYSTEM-PROJECT-FOR-MICRO-IT-

Overview

This Student Management System (SMS) is a web application designed to help educational institutions efficiently manage student data and administrative tasks. It incorporates AI features to predict student performance and identify students at risk of underperforming, alongside traditional functionalities such as enrollment, attendance tracking, grade management, and parent-teacher communication.

##Features

Student Enrollment: Add new students with basic details (name, ID, branch).

Attendance Tracking: Record and display student attendance.

Grade Management: Manage and record student grades using letter grades (A, B, C, D, E, F).

Performance Prediction: AI-powered prediction of final exam scores based on semester marks, attendance, and activities participation.

At-Risk Detection: Identify students who may be at risk of poor academic performance.

Parent-Teacher Interaction: Facilitate communication through messaging between parents and teachers.

Data Visualization: Display distribution of final exam scores and feature importance in predictions.

##Technologies Used

Python 3

Streamlit (for web app interface)

Pandas, NumPy (data manipulation)

Scikit-learn (machine learning models)

Matplotlib (data visualization)

Pickle (model serialization)

##Installation & Setup
##Clone the repository:

git clone <repository-url>
cd student-management-system

##Install dependencies:

pip install -r requirements.txt
Prepare dataset:
Place your students.csv dataset in the specified folder or update the path in app.py.

Train models (optional if models already provided):
Run the training script to generate the scaler and models:

python train_models.py
##Run the Streamlit app:

streamlit run app.py
Usage
Enroll students using the enrollment form.

Input academic details and attendance for performance prediction.

Track attendance and manage grades.

Send and view messages between parents and teachers.

Project Structure
graphql
├── app.py                  # Main Streamlit application
├── train_models.py         # Script to train and save ML models
├── models/                 # Folder containing saved models and scaler
│   ├── scaler.pkl
│   ├── performance_model.pkl
│   └── at_risk_model.pkl
├── students.csv            # Dataset of student records
├── students_enrolled.csv   # CSV file where enrolled student details are saved
├── parent_teacher_messages.csv  # CSV file for parent-teacher messages
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


##Future Enhancements
Add role-based authentication (admin, teacher, parent, student).

Integrate real-time attendance via biometric or RFID systems.

Include detailed analytics dashboards.

Mobile app interface for easier access.

Automate notifications and alerts for at-risk students and parents.

