import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

model = joblib.load('student_score_model.pkl')
st.title("Student Exam Score Prediction ")
st.write("Enter your grade and predicate your grade ")
hours = st.number_input("study hours  ", min_value=0.0, max_value=50.0, step=0.5)
if st.button("Predicate your grade "):
    prediction = model.predict([[hours]])[0]
    st.success(f" predicated grade : {prediction:.2f}")
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Actual_Score":  [50, 52, 53, 55, 58, 60, 62, 63, 65, 67, 70]
}

df = pd.DataFrame(data)

df["Predicted_Score"] = model.predict(df[["Hours_Studied"]])


fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Hours_Studied", y="Actual_Score", label="Actual", ax=ax)
sns.lineplot(data=df, x="Hours_Studied", y="Predicted_Score", label="Predicted", ax=ax, color="orange")
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.legend()

st.pyplot(fig)
