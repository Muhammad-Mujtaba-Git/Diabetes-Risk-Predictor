import streamlit as st
import requests

st.title("🩺 Diabetes Risk Checker")
st.write("Answer a few simple questions to check your diabetes risk.")
st.warning("⚠️ This is a screening tool only. Not a medical diagnosis. Always consult a doctor.")

st.subheader("About You")
gender = st.selectbox("What is your gender?", ["Male", "Female"])
region = st.selectbox("Where do you live?", ["Urban", "Rural"])
family_history = st.selectbox("Does anyone in your family have diabetes?", ["No", "Yes"])

st.subheader("Your Body Measurements")
weight = st.number_input("Your weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Your height (cm)", min_value=100.0, max_value=220.0, value=165.0)
waist = st.number_input("Your waist size (inches)", min_value=20.0, max_value=70.0, value=35.0)

st.subheader("Your Blood Pressure")
st.write("You can find this on your last medical report.")
systolic = st.number_input("Upper number (Systolic)", min_value=80, max_value=250, value=120)
diastolic = st.number_input("Lower number (Diastolic)", min_value=40, max_value=150, value=80)

st.subheader("Blood Test Results")
hdl = st.number_input("Good Cholesterol (HDL)", min_value=10.0, max_value=100.0, value=50.0)

st.subheader("Your Lifestyle")
thirst = st.selectbox("Do you feel unusually thirsty lately?", ["No", "Yes"])
urination = st.selectbox("Do you urinate more than usual?", ["No", "Yes"])
exercise = st.number_input("How many hours do you exercise per day?", min_value=0.0, max_value=5.0, value=0.5)

if st.button("Check My Risk"):
    payload = {
        "gender": gender,
        "region": region,
        "weight": weight,
        "height": height,
        "waist": waist,
        "systolic": systolic,
        "diastolic": diastolic,
        "family_history": family_history,
        "thirst": thirst,
        "urination": urination,
        "hdl": hdl,
        "exercise_hours": exercise
    }
    
    try:
        response = requests.post("https://muhammadmujtabaaiml-diabetes-risk-predictor.hf.space/predict", json=payload)
        result = response.json()
        prob = float(result['data']['probability'].replace('%',''))
        diagnosis = result['data']['diagnosis']
        bmi = result['data']['details']['computed_bmi']
        
        st.subheader("Your Results")
        st.metric("Risk Score", f"{prob:.1f}%")
        st.metric("Your BMI", bmi)
        
        if prob >= 65:
            st.error(f"🔴 {diagnosis} — Please consult a doctor soon.")
        elif prob >= 35:
            st.warning(f"🟡 Borderline Risk — Monitor your health and see a doctor.")
        else:
            st.success(f"🟢 {diagnosis} — Keep up your healthy habits!")
            
        st.info("Remember: This tool screens for risk only. A doctor's diagnosis is always required.")
        st.divider()
        with st.expander("🔍 See what affected your result"):
            with st.spinner("Analyzing model signals..."):
                exp_response = requests.post("https://muhammadmujtabaaiml-diabetes-risk-predictor.hf.space/explain", json=payload)
                explanation = exp_response.json()["explanation"]

                for item in explanation:
                    # Only show features that actually had an impact
                    if abs(item["shap_value"]) > 0:
                        icon = "🔺" if item["impact"] == "Increases Risk" else "🔹"
                        st.write(f"{icon} **{item['feature']}** ({item['impact']})")
            
    except Exception as e:
        st.error("Connection to AI service failed. Please check if the Hugging Face space is active.")
        # ── END ───────────────────────────────────────────────────────────────
        
    except:
        st.error("Service temporarily unavailable. Please try again later.")