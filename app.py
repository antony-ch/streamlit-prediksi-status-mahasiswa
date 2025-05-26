import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan komponen pendukung
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')
selected_features_mask = joblib.load('selected_features_mask.pkl')
fitur_awal = joblib.load('fitur_awal.pkl')
label_mapping = joblib.load('label_mapping.pkl')

st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("""
Aplikasi ini memprediksi status mahasiswa berdasarkan data mahasiswa dan akademik mereka.  
Silakan isi form di bawah ini dan klik tombol **Prediksi Status**.
""")

with st.form("form_prediksi"):
    st.header("Formulir Data Mahasiswa")

    col1, col2, col3 = st.columns(3)

    with col1:
        marital_status = st.selectbox("Status Pernikahan", ['Single', 'Married', 'Widower', 'Divorced', 'Fakto Union', 'Legally Separated'])
        application_mode = st.selectbox("Metode Pendaftaran", [
    "1st phase - general contingent",
    "Ordinance No. 612/93",
    "1st phase - special contingent (Azores Island)",
    "Holders of other higher courses",
    "Ordinance No. 854-B/99",
    "International student (bachelor)",
    "1st phase - special contingent (Madeira Island)",
    "2nd phase - general contingent",
    "3rd phase - general contingent",
    "Ordinance No. 533-A/99, item b2) (Different Plan)",
    "Ordinance No. 533-A/99, item b3 (Other Institution)",
    "Over 23 years old",
    "Transfer",
    "Change of course",
    "Technological specialization diploma holders",
    "Change of institution/course",
    "Short cycle diploma holders",
    "Change of institution/course (International)"
])
        program_studi = st.selectbox("Program Studi", ["Biofuel Production Technologies",
    "Animation and Multimedia Design",
    "Social Service (evening attendance)",
    "Agronomy",
    "Communication Design",
    "Veterinary Nursing",
    "Informatics Engineering",
    "Equinculture",
    "Management",
    "Social Service",
    "Tourism",
    "Nursing",
    "Oral Hygiene",
    "Advertising and Marketing Management",
    "Journalism and Communication",
    "Basic Education",
    "Management (evening attendance)"])
        waktu_belajar = st.selectbox("Waktu Belajar", ['Daytime', 'Evening'])
        kualifikasi_sebelumnya = st.selectbox("Kualifikasi Sebelumnya", [ "Secondary education",
    "Higher education - bachelor's degree",
    "Higher education - degree",
    "Higher education - master's",
    "Higher education - doctorate",
    "Frequency of higher education",
    "12th year of schooling - not completed",
    "11th year of schooling - not completed",
    "Other - 11th year of schooling",
    "10th year of schooling",
    "10th year of schooling - not completed",
    "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    "Technological specialization course",
    "Higher education - degree (1st cycle)",
    "Professional higher technical course",
    "Higher education - master (2nd cycle)"])
        kebangsaan = st.selectbox("Kebangsaan", ["Portuguese", "German", "Spanish", "Italian", "Dutch", "English", "Lithuanian",
    "Angolan", "Cape Verdean", "Guinean", "Mozambican", "Santomean", "Turkish",
    "Brazilian", "Romanian", "Moldova (Republic of)", "Mexican", "Ukrainian",
    "Russian", "Cuban", "Colombian"])

    with col2:
        pendidikan_ibu = st.selectbox("Pendidikan Ibu", ["Secondary Education - 12th Year of Schooling or Eq.",
    "Higher Education - Bachelor's Degree",
    "Higher Education - Degree",
    "Higher Education - Master's",
    "Higher Education - Doctorate",
    "Frequency of Higher Education",
    "12th Year of Schooling - Not Completed",
    "11th Year of Schooling - Not Completed",
    "7th Year (Old)",
    "Other - 11th Year of Schooling",
    "10th Year of Schooling",
    "General commerce course",
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Eq.",
    "Technical-professional course",
    "7th year of schooling",
    "2nd cycle of the general high-school course",
    "9th Year of Schooling - Not Completed",
    "8th year of schooling",
    "Unknown",
    "Can't read or write",
    "Can read without having a 4th year of schooling",
    "Basic Education 1st Cycle (4th/5th Year) or Eq.",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Eq.",
    "Technological specialization course",
    "Higher education - degree (1st cycle)",
    "Specialized higher studies course",
    "Professional higher technical course",
    "Higher Education - Master (2nd cycle)",
    "Higher Education - Doctorate (3rd cycle)"])
        pendidikan_ayah = st.selectbox("Pendidikan Ayah", ["Secondary Education - 12th Year of Schooling or Eq.",
    "Higher Education - Bachelor's Degree",
    "Higher Education - Degree",
    "Higher Education - Master's",
    "Higher Education - Doctorate",
    "Frequency of Higher Education",
    "12th Year of Schooling - Not Completed",
    "11th Year of Schooling - Not Completed",
    "7th Year (Old)",
    "Other - 11th Year of Schooling",
    "2nd year complementary high-school course",
    "10th Year of Schooling",
    "General commerce course",
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Eq.",
    "Complementary High-School Course",
    "Technical-professional course",
    "Complementary High-School Course - not concluded",
    "7th year of schooling",
    "2nd cycle of the general high-school course",
    "9th Year of Schooling - Not Completed",
    "8th year of schooling",
    "General Course of Administration and Commerce",
    "Supplementary Accounting and Administration",
    "Unknown",
    "Can't read or write",
    "Can read without having a 4th year of schooling",
    "Basic Education 1st Cycle (4th/5th Year) or Eq.",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Eq.",
    "Technological specialization course",
    "Higher education - degree (1st cycle)",
    "Specialized higher studies course",
    "Professional higher technical course",
    "Higher Education - Master (2nd cycle)",
    "Higher Education - Doctorate (3rd cycle)"])
        pekerjaan_ibu = st.selectbox("Pekerjaan Ibu", ["Student",
    "Representatives of Legislative & Executive Bodies, Directors and Executive Managers",
    "Specialists in Intellectual and Scientific Activities",
    "Intermediate Level Technicians and Professions",
    "Administrative staff",
    "Personal Services, Security and Safety Workers and Sellers",
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    "Skilled Workers in Industry, Construction and Craftsmen",
    "Installation and Machine Operators and Assembly Workers",
    "Unskilled Workers",
    "Armed Forces Professions",
    "Other Situation",
    "(blank)",
    "Health professionals",
    "Teachers",
    "Specialists in information and communication technologies (ICT)",
    "Intermediate level science and engineering technicians and professions",
    "Technicians and professionals, of intermediate level of health",
    "Intermediate level technicians from legal, social, sports, cultural and similar services",
    "Office workers, secretaries in general and data processing operators",
    "Data, accounting, statistical, financial services and registry-related operators",
    "Other administrative support staff",
    "Personal service workers",
    "Sellers",
    "Personal care workers and the like",
    "Skilled construction workers and the like, except electricians",
    "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like",
    "Workers in food processing, woodworking, clothing and other industries and crafts",
    "Cleaning workers",
    "Unskilled workers in agriculture, animal production, fisheries and forestry",
    "Unskilled workers in extractive industry, construction, manufacturing and transport",
    "Meal preparation assistants"])
        pekerjaan_ayah = st.selectbox("Pekerjaan Ayah", [ "Student",
    "Representatives of Legislative & Executive Bodies, Directors and Executive Managers",
    "Specialists in Intellectual and Scientific Activities",
    "Intermediate Level Technicians and Professions",
    "Administrative staff",
    "Personal Services, Security and Safety Workers and Sellers",
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    "Skilled Workers in Industry, Construction and Craftsmen",
    "Installation and Machine Operators and Assembly Workers",
    "Unskilled Workers",
    "Armed Forces Professions",
    "Other Situation",
    "(blank)",
    "Armed Forces Officers",
    "Armed Forces Sergeants",
    "Other Armed Forces personnel",
    "Directors of administrative and commercial services",
    "Hotel, catering, trade and other services directors",
    "Specialists in the physical sciences, mathematics, engineering and related techniques",
    "Health professionals",
    "Teachers",
    "Specialists in finance, accounting, administrative organization, public and commercial relations",
    "Intermediate level science and engineering technicians and professions",
    "Technicians and professionals, of intermediate level of health",
    "Intermediate level technicians from legal, social, sports, cultural and similar services",
    "Information and communication technology technicians",
    "Office workers, secretaries in general and data processing operators",
    "Data, accounting, statistical, financial services and registry-related operators",
    "Other administrative support staff",
    "Personal service workers",
    "Sellers",
    "Personal care workers and the like",
    "Protection and security services personnel",
    "Market-oriented farmers and skilled agricultural and animal production workers",
    "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence",
    "Skilled construction workers and the like, except electricians",
    "Skilled workers in metallurgy, metalworking and similar",
    "Skilled workers in electricity and electronics",
    "Workers in food processing, woodworking, clothing and other industries and crafts",
    "Fixed plant and machine operators",
    "Assembly workers",
    "Vehicle drivers and mobile equipment operators",
    "Unskilled workers in agriculture, animal production, fisheries and forestry",
    "Unskilled workers in extractive industry, construction, manufacturing and transport",
    "Meal preparation assistants",
    "Street vendors (except food) and street service providers"])
        mahasiswa_displaced = st.selectbox("Mahasiswa Displaced?", ['Yes', 'No'])
        kebutuhan_khusus = st.selectbox("Berkebutuhan Khusus?", ['Yes', 'No'])

    with col3:
        options_tunggakan = {'Yes': 1, 'No': 0}
        choice_tunggakan = st.selectbox("Memiliki Tunggakan?", list(options_tunggakan.keys()))
        memiliki_tunggakan = options_tunggakan[choice_tunggakan]

        options_lunas = {'Yes': 1, 'No': 0}
        choice = st.selectbox("Pembayaran Lunas?", list(options_lunas.keys()))
        pembayaran_lunas = options_lunas[choice]

        jenis_kelamin = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
        beasiswa = st.selectbox("Penerima Beasiswa?", ['Yes', 'No'])
        internasional = st.selectbox("Mahasiswa Internasional?", ['Yes', 'No'])
        usia = st.number_input("Usia Saat Masuk", min_value=17, max_value=70, value=20)
        nilai_penerimaan = st.number_input("Nilai Penerimaan", min_value=0.0, max_value=200.0, value=150.0)

    st.subheader("Data Akademik Semester")

    col4, col5 = st.columns(2)
    with col4:
        cu1_enrolled = st.number_input("Jumlah Mata Kuliah Semester 1", min_value=0, value=6)
        cu1_approved = st.number_input("Jumlah Lulus Semester 1", min_value=0, value=5)
        cu1_evaluations = st.number_input("Jumlah Evaluasi Semester 1", min_value=0, value=6)
        cu1_grade = st.number_input("Rata-rata Nilai Semester 1", min_value=0.0, max_value=20.0, value=13.5)
        cu1_credited = st.number_input("Jumlah Kredit Transfer Semester 1", min_value=0, value=0)
        cu1_wo_eval = st.number_input("Tanpa Evaluasi Semester 1", min_value=0, value=0)

    with col5:
        cu2_enrolled = st.number_input("Jumlah Mata Kuliah Semester 2", min_value=0, value=5)
        cu2_approved = st.number_input("Jumlah Lulus Semester 2", min_value=0, value=4)
        cu2_evaluations = st.number_input("Jumlah Evaluasi Semester 2", min_value=0, value=5)
        cu2_grade = st.number_input("Rata-rata Nilai Semester 2", min_value=0.0, max_value=20.0, value=14.0)
        cu2_credited = st.number_input("Jumlah Kredit Transfer Semester 2", min_value=0, value=0)
        cu2_wo_eval = st.number_input("Tanpa Evaluasi Semester 2", min_value=0, value=0)

    submit = st.form_submit_button("🔍 Prediksi Status")

if submit:
    data_input = {
        'Marital_status': marital_status,
        'Application_mode': application_mode,
        'Course': program_studi,
        'Daytime_evening_attendance': waktu_belajar,
        'Previous_qualification': kualifikasi_sebelumnya,
        'Nacionality': kebangsaan,
        'Mothers_qualification': pendidikan_ibu,
        'Fathers_qualification': pendidikan_ayah,
        'Mothers_occupation': pekerjaan_ibu,
        'Fathers_occupation': pekerjaan_ayah,
        'Displaced': mahasiswa_displaced,
        'Educational_special_needs': kebutuhan_khusus,
        'Debtor': memiliki_tunggakan,
        'Tuition_fees_up_to_date': pembayaran_lunas,
        'Gender': jenis_kelamin,
        'Scholarship_holder': beasiswa,
        'International': internasional,
        'Age_at_enrollment': usia,
        'Admission_grade': nilai_penerimaan,
        'Curricular_units_1st_sem_credited': cu1_credited,
        'Curricular_units_1st_sem_enrolled': cu1_enrolled,
        'Curricular_units_1st_sem_evaluations': cu1_evaluations,
        'Curricular_units_1st_sem_approved': cu1_approved,
        'Curricular_units_1st_sem_grade': cu1_grade,
        'Curricular_units_1st_sem_without_evaluation': cu1_wo_eval,
        'Curricular_units_2nd_sem_credited': cu2_credited,
        'Curricular_units_2nd_sem_enrolled': cu2_enrolled,
        'Curricular_units_2nd_sem_evaluations': cu2_evaluations,
        'Curricular_units_2nd_sem_approved': cu2_approved,
        'Curricular_units_2nd_sem_grade': cu2_grade,
        'Curricular_units_2nd_sem_without_evaluation': cu2_wo_eval
    }

    df_input = pd.DataFrame([data_input])

    # Feature Engineering tambahan
    df_input['Approved_Ratio_1st'] = df_input['Curricular_units_1st_sem_approved'] / (df_input['Curricular_units_1st_sem_enrolled'] + 1e-5)
    df_input['Failures_1st'] = df_input['Curricular_units_1st_sem_enrolled'] - df_input['Curricular_units_1st_sem_approved']
    df_input['Debt_Tuition_Interaction'] = df_input['Debtor'] * df_input['Tuition_fees_up_to_date']

    kategorikal = ['Marital_status', 'Application_mode', 'Course', 'Daytime_evening_attendance',
                   'Previous_qualification', 'Nacionality', 'Mothers_qualification', 'Fathers_qualification',
                   'Mothers_occupation', 'Fathers_occupation', 'Displaced', 'Educational_special_needs',
                   'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'International']

    df_encoded = pd.get_dummies(df_input, columns=kategorikal, drop_first=True)

    for col in fitur_awal:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[fitur_awal]

    df_scaled = scaler.transform(df_encoded)
    df_selected = df_scaled[:, selected_features_mask]

    prediksi = model.predict(df_selected)[0]
    hasil = label_mapping[prediksi]

    st.success(f"Status Mahasiswa yang Diprediksi: **{hasil}**")