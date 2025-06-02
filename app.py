import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import datetime

st.title("🎓 AI-Powered Student Management System")

# -----------------------------
# 1. Student Enrollment Section
# -----------------------------
st.subheader("📋 Enroll a New Student")
name = st.text_input("Student Name")
student_id = st.text_input("Student ID")
branch = st.selectbox("Branch", ["CSE", "ECE", "EEE", "MECH", "CIVIL"])

if st.button("Save Student"):
    new_student = pd.DataFrame([[name, student_id, branch]], columns=["Name", "Student_ID", "Branch"])
    new_student.to_csv("students_enrolled.csv", mode='a', header=not os.path.exists("students_enrolled.csv"), index=False)
    st.success("✅ Student enrolled successfully!")

# -----------------------------
# 2. Display Enrolled Students
# -----------------------------
st.subheader("📑 Enrolled Students")
if os.path.exists("students_enrolled.csv"):
    enrolled_data = pd.read_csv("students_enrolled.csv")
    st.dataframe(enrolled_data)
else:
    st.warning("⚠️ No students enrolled yet.")

# -----------------------------
# 3. Attendance Tracking Section
# -----------------------------
st.subheader("🗓️ Attendance Tracking")

if os.path.exists("students_enrolled.csv"):
    student_list = enrolled_data['Name'].tolist()
else:
    student_list = []

date = st.date_input("Select Date", datetime.date.today())
student = st.selectbox("Select Student", student_list)
status = st.radio("Attendance Status", ['Present', 'Absent'])

if st.button("Mark Attendance"):
    attendance_record = pd.DataFrame([[date, student, status]], columns=["Date", "Student", "Status"])
    attendance_record.to_csv("attendance.csv", mode='a', header=not os.path.exists("attendance.csv"), index=False)
    st.success(f"Attendance marked as {status} for {student} on {date}")

st.subheader("📋 View Attendance Records")
if os.path.exists("attendance.csv"):
    attendance_data = pd.read_csv("attendance.csv")
    st.dataframe(attendance_data)
else:
    st.info("No attendance records yet.")

    # -----------------------------
# -----------------------------
# 4. Grade Management Section (Letter Grades)
# -----------------------------
st.subheader("📝 Grade Management")

if os.path.exists("students_enrolled.csv"):
    student_list = enrolled_data['Name'].tolist()
else:
    student_list = []

grade_student = st.selectbox("Select Student for Grades", student_list, key="grade_student")
grade_semester = st.selectbox("Select Semester", ["Sem1", "Sem2", "Sem3"])
grade_value = st.selectbox("Select Grade", ['A', 'B', 'C', 'D', 'E', 'F'], key="grade_value")

if st.button("Save Grade"):
    # Load or create grades dataframe
    if os.path.exists("grades.csv"):
        grades_df = pd.read_csv("grades.csv")
    else:
        grades_df = pd.DataFrame(columns=["Student", "Semester", "Grade"])
    
    # Update existing grade if present
    existing_idx = grades_df[(grades_df['Student'] == grade_student) & (grades_df['Semester'] == grade_semester)].index
    if len(existing_idx) > 0:
        grades_df.loc[existing_idx[0], 'Grade'] = grade_value
        st.info(f"Updated {grade_semester} grade for {grade_student} to {grade_value}.")
    else:
        new_grade = pd.DataFrame([[grade_student, grade_semester, grade_value]], columns=["Student", "Semester", "Grade"])
        grades_df = pd.concat([grades_df, new_grade], ignore_index=True)
        st.success(f"Added {grade_semester} grade {grade_value} for {grade_student}.")
    
    grades_df.to_csv("grades.csv", index=False)

st.subheader("📋 View All Grades")
if os.path.exists("grades.csv"):
    grades_data = pd.read_csv("grades.csv")
    st.dataframe(grades_data)
else:
    st.info("No grades recorded yet.")

# -----------------------------
# 4. Load Dataset for Visualization
# -----------------------------
data = pd.read_csv('C:/Users/prath/OneDrive/Documents/Desktop/student mangement system/students.csv')  # <-- Change path if needed

# Load models and scaler
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
reg_model = pickle.load(open('models/performance_model.pkl', 'rb'))
cls_model = pickle.load(open('models/at_risk_model.pkl', 'rb'))

features = ['Sem1', 'Sem2', 'Sem3', 'Attendance', 'Activities']

# -----------------------------
# 5. Feature Importance
# -----------------------------
st.subheader("📊 Feature Importance for At-Risk Prediction")
feature_importance = dict(zip(features, cls_model.coef_[0]))
for feat, imp in feature_importance.items():
    st.write(f"{feat}: {imp:.3f}")

# -----------------------------
# 6. Histogram of Final Scores
# -----------------------------
st.subheader("📈 Distribution of Final Exam Scores")
fig, ax = plt.subplots()
ax.hist(data['Final Exam Score'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel('Final Exam Score')
ax.set_ylabel('Number of Students')
st.pyplot(fig)

# -----------------------------
# 7. Prediction Input
# -----------------------------
st.subheader("🔍 Enter Student Academic Details")
Sem1 = st.number_input('Semester 1 Marks', min_value=0, max_value=100)
Sem2 = st.number_input('Semester 2 Marks', min_value=0, max_value=100)
Sem3 = st.number_input('Semester 3 Marks', min_value=0, max_value=100)
Attendance = st.number_input('Attendance (%)', min_value=0, max_value=100)
Activities = st.selectbox('Participated in Activities?', ['Yes', 'No'])

if st.button('Predict'):
    act = 1 if Activities == 'Yes' else 0
    input_data = pd.DataFrame([[Sem1, Sem2, Sem3, Attendance, act]], columns=features)
    input_scaled = scaler.transform(input_data)

    # Predict final exam score (regression)
    pred_score = reg_model.predict(input_scaled)[0]

    # Predict at-risk status and probability (classification)
    at_risk = cls_model.predict(input_scaled)[0]
    at_risk_prob = cls_model.predict_proba(input_scaled)[0][1]

    st.write(f"**Predicted Final Score:** {pred_score:.2f}")
    st.write(f"**At Risk:** {'Yes' if at_risk == 1 else 'No'}")
    st.write(f"**At Risk Probability:** {at_risk_prob:.2%}")

# -----------------------------
# 8. Parent-Teacher Interaction
# -----------------------------
st.subheader("👨‍👩‍👧‍👦 Parent-Teacher Interaction")
parent_name = st.text_input("Parent Name")
teacher_name = st.text_input("Teacher Name")
message = st.text_area("Message")

if st.button("Send Message"):
    interaction = pd.DataFrame([[parent_name, teacher_name, message]], columns=["Parent", "Teacher", "Message"])
    interaction.to_csv("parent_teacher_messages.csv", mode='a', header=not os.path.exists("parent_teacher_messages.csv"), index=False)
    st.success("📩 Message sent successfully!")

# Display existing messages
st.subheader("📬 Previous Messages")
if os.path.exists("parent_teacher_messages.csv"):
    msgs = pd.read_csv("parent_teacher_messages.csv")
    st.dataframe(msgs)
else:
    st.info("No messages yet.")
