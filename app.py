import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. خلق البيانات وتدريب النموذج
# ==========================================
@st.cache_resource
def train_model():
    np.random.seed(42)
    random.seed(42)
    data = []
    for i in range(1, 501): # 500 تلميذ
        groupe = random.choice(['A', 'B', 'C'])
        if groupe == 'A':
            base_note = 15; absences = np.random.randint(0, 5); participation = np.random.randint(3, 6); comportement = np.random.randint(4, 6)
        elif groupe == 'B':
            base_note = 11; absences = np.random.randint(3, 10); participation = np.random.randint(2, 5); comportement = np.random.randint(2, 5)
        else:
            base_note = 7; absences = np.random.randint(8, 20); participation = np.random.randint(1, 4); comportement = np.random.randint(1, 4)
        
        notes = [np.clip(base_note + np.random.normal(0, 1.5), 0, 20) for _ in range(6)]
        devoirs = random.choice(['Oui', 'Non']); activites = random.choice(['Oui', 'Non'])
        
        score = 0
        if absences < 5: score += 2
        if participation >= 4: score += 1
        if comportement >= 4: score += 1
        if devoirs == 'Oui': score += 2
        if activites == 'Oui': score += 1
        if notes[-1] > notes[0]: score += 2 
        
        if absences > 10: score -= 3
        if participation <= 2: score -= 1
        if notes[-1] < notes[0]: score -= 2 
        
        if score >= 4: evolution = 'صعود (Upgrade)'
        elif score <= -2: evolution = 'نزول (Downgrade)'
        else: evolution = 'بقاء (Stable)'
            
        data.append([groupe, notes[0], notes[1], notes[2], notes[3], notes[4], notes[5], absences, participation, comportement, devoirs, activites, evolution])
        
    columns = ['groupe', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'absences', 'participation', 'comportement', 'devoirs', 'activites', 'evolution']
    df = pd.DataFrame(data, columns=columns)
    
    df['groupe'] = df['groupe'].map({'A': 2, 'B': 1, 'C': 0})
    df['devoirs'] = df['devoirs'].map({'Oui': 1, 'Non': 0})
    df['activites'] = df['activites'].map({'Oui': 1, 'Non': 0})
    
    X = df.drop('evolution', axis=1)
    y = df['evolution']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ==========================================
# 2. واجهة المستخدم
# ==========================================
st.set_page_config(page_title="توقع مستوى التلميذ", layout="centered")
st.title("🎓 نظام التنبؤ بالمستوى الدراسي")
st.write("أدخل معلومات التلميذ، وسيقوم الذكاء الاصطناعي بالتنبؤ بمستواه في الفصل القادم.")

with st.form("student_form"):
    st.subheader("معلومات التلميذ الحالية")
    col1, col2 = st.columns(2)
    with col1:
        groupe = st.selectbox("الفئة الحالية (Groupe):", ['A', 'B', 'C'])
        absences = st.number_input("عدد الغيابات:", min_value=0, max_value=50, value=5)
        participation = st.slider("المشاركة (1-5):", 1, 5, 3)
    with col2:
        comportement = st.slider("السلوك (1-5):", 1, 5, 3)
        devoirs = st.selectbox("هل ينجز الواجبات المنزلية؟", ['Oui', 'Non'])
        activites = st.selectbox("هل يشارك في الأنشطة؟", ['Oui', 'Non'])
        
    st.subheader("نقاط الفصول الستة السابقة")
    col3, col4, col5 = st.columns(3)
    with col3:
        n1 = st.number_input("نقطة الفصل 1:", 0.0, 20.0, 12.0)
        n2 = st.number_input("نقطة الفصل 2:", 0.0, 20.0, 12.0)
    with col4:
        n3 = st.number_input("نقطة الفصل 3:", 0.0, 20.0, 12.0)
        n4 = st.number_input("نقطة الفصل 4:", 0.0, 20.0, 12.0)
    with col5:
        n5 = st.number_input("نقطة الفصل 5:", 0.0, 20.0, 12.0)
        n6 = st.number_input("نقطة الفصل 6:", 0.0, 20.0, 12.0)
        
    submit = st.form_submit_button(" توقع مستوى التلميذ")

# ==========================================
# 3. عرض النتيجة
# ==========================================
if submit:
    input_data = pd.DataFrame([[
        {'A': 2, 'B': 1, 'C': 0}[groupe],
        n1, n2, n3, n4, n5, n6,
        absences, participation, comportement,
        1 if devoirs == 'Oui' else 0,
        1 if activites == 'Oui' else 0
    ]], columns=model.feature_names_in_)
    
    prediction = model.predict(input_data)[0]
    
    st.markdown("---")
    st.subheader("النتيجة:")
    if "صعود" in prediction:
        st.success(f" التلميذ سيشهد تحسن **{prediction}** في الفصل القادم. (مجهود ممتاز!)")
    elif "نزول" in prediction:
        st.error(f"التلميذ معرض للخطر **{prediction}**. يرجى التدخل ومتابعته!")
    else:
        st.info(f" التلميذ سيحافظ على مستواه الحالي **{prediction}**.")
