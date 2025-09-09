import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import pickle
import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(local_image_path):
    encoded = get_base64(local_image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        padding-top: 60px;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.68); 
        padding: 2.5rem;
        border-radius: 18px;
        box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.3);
        width: 70%;
        margin-top: 20vh;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background
set_background("thermal power plant.jpg")

st.markdown("""
    <style>
    /* Make all headings black */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: black !important;
    }

    /* Make label texts black (Select Power Station, etc.) */
    label, .stApp [data-testid="stSelectboxLabel"] {
        color: black !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- MongoDB Setup ----------
client = MongoClient("mongodb://localhost:27017/")
db = client["tripping_analysis"]
collection = db["unit_tripping_logs"]

# Clean field names once
for doc in collection.find():
    cleaned = {k.strip(): v for k, v in doc.items()}
    collection.update_one({'_id': doc['_id']}, {'$set': cleaned})

# ---------- Load ML Model ----------
with open("tripping_model.pkl", "rb") as f:
    model, le_station, le_unit, le_reason = pickle.load(f)


# ---------- Helper Functions ----------
def try_parse_db_date(date_str):
    formats = ["%d/%m/%Y %H:%M", "%d/%m/%y %H:%M", "%#d/%#m/%Y %H:%M", "%-d/%-m/%Y %H:%M"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None


def format_display(dt):
    return dt.strftime("%d/%m/%Y %H:%M") if dt else "Invalid"


# ---------- Session State Setup ----------
if "page" not in st.session_state:
    st.session_state.page = 1

# ---------- Page 1: Station and Unit Selection ----------
if st.session_state.page == 1:
    st.set_page_config(page_title="Tripping Analysis", layout="centered")
    st.title("⚡ CSPGCL Tripping Analyzer")
    st.subheader("Start your analysis by entering the basic plant details below.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        stations = sorted(collection.distinct("Power Station"))
        selected_station = st.selectbox("Select Power Station", stations)

        units = sorted(collection.find({"Power Station": selected_station}).distinct("Unit"))
        selected_unit = st.selectbox("Select Unit", units)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        if st.button("➡️ Next"):
            st.session_state.station = selected_station
            st.session_state.unit = selected_unit
            st.session_state.page = 2
            st.rerun()

# ---------- Page 2: Input Time and Analyze ----------
elif st.session_state.page == 2:
    st.title("Enter Date/Time and Analyze")
    st.write(f"**Selected Station:** {st.session_state.station}")
    st.write(f"**Selected Unit:** {st.session_state.unit}")

    st.markdown("### 🔄 Tripping Date/Time ")
    col1, col2 = st.columns([2, 2])
    with col1:
        last_date = st.date_input("Tripping Date")
    with col2:
        last_time = st.time_input("Tripping Time")
    last_tripping = datetime.combine(last_date, last_time)

    st.markdown("### 🔄 Lit-up Date/Time ")
    nil_lit = st.checkbox("NIL")
    lit_up_time = None
    if not nil_lit:
        col1, col2 = st.columns([2, 2])
        with col1:
            lit_date = st.date_input("Lit-Up Date")
        with col2:
            lit_time = st.time_input("Lit-Up Time")
        lit_up_time = datetime.combine(lit_date, lit_time)

    st.markdown("### 🔄 Synchronization Date/Time ")
    nil_sync = st.checkbox("Nil")
    sync_time = None
    if not nil_sync:
        col1, col2 = st.columns([2, 2])
        with col1:
            sync_date = st.date_input("Synchronization Date")
        with col2:
            sync_time_input = st.time_input("Synchronization Time")
        sync_time = datetime.combine(sync_date, sync_time_input)

    # ------- Prepare features for anomaly detection -------
    gap_lit = (lit_up_time - last_tripping).total_seconds() / 60 if lit_up_time else -1
    gap_sync = (sync_time - last_tripping).total_seconds() / 60 if sync_time else -1

    station_enc = le_station.transform([st.session_state.station])[0]
    unit_enc = le_unit.transform([st.session_state.unit])[0]

    input_features = [[station_enc, unit_enc, gap_lit, gap_sync]]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Analyze"):
            candidates = collection.find({
                "Power Station": st.session_state.station,
                "Unit": st.session_state.unit
            })

            found = None
            for doc in candidates:
                db_last = try_parse_db_date(doc.get("Last Tripping", ""))
                db_lit = try_parse_db_date(doc.get("Lit-up time", ""))
                db_sync = try_parse_db_date(doc.get("Synchronising", ""))

                match_last = (db_last == last_tripping)
                match_lit = (nil_lit or db_lit == lit_up_time)
                match_sync = (nil_sync or db_sync == sync_time)

                if match_last and match_lit and match_sync:
                    found = doc
                    break

            if found:
                reason = found.get("Reason", "❌ No Reason Found")
                status = "✅ Match Found"
            else:
                # -------- ML Prediction if No Match --------
                try:
                    station_enc = le_station.transform([st.session_state.station])[0]
                    unit_enc = le_unit.transform([st.session_state.unit])[0]
                    gap_lit = -1 if nil_lit or not lit_up_time else (lit_up_time - last_tripping).total_seconds() / 60
                    gap_sync = -1 if nil_sync or not sync_time else (sync_time - last_tripping).total_seconds() / 60

                    X_pred = [[station_enc, unit_enc, gap_lit, gap_sync]]
                    pred_encoded = model.predict(X_pred)[0]
                    predicted_reason = le_reason.inverse_transform([pred_encoded])[0]
                    reason = f"ML Predicted: {predicted_reason}"
                except Exception as e:
                    reason = "⚠️ ML Prediction Failed"
                status = "⚠️ No Match"

            log_data = {
                "Power Station": st.session_state.station,
                "Unit": st.session_state.unit,
                "Last Tripping Time": format_display(last_tripping),
                "Lit-up Time": "NIL" if nil_lit else format_display(lit_up_time),
                "Synchronization Time": "NIL" if nil_sync else format_display(sync_time),
                "Reason": reason
            }

            st.session_state.analysis_result = {
                "log_data": log_data,
                "status": status,
                "success": bool(found)
            }

            st.session_state.page = 3
            st.rerun()

# ---------- Page 3: Show Enhanced Result ----------
elif st.session_state.page == 3:
    result = st.session_state.get("analysis_result", {})
    log_data = result.get("log_data", {})
    status = result.get("status", "No Result")
    success = result.get("success", False)

    st.title("📊 Result")

    status_color = "green" if success else "red"
    st.markdown(f"<h3 style='color:{status_color};'>{status}</h3>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 25px 35px; border-radius: 12px;
                    border: 1px solid #ddd; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); width: 60%; margin: auto;">
            <p><strong>🏭 Power Station:</strong> {log_data.get("Power Station", "N/A")}</p>
            <p><strong>🛠️ Unit:</strong> {log_data.get("Unit", "N/A")}</p>
            <p><strong>🔁 Last Tripping:</strong> {log_data.get("Last Tripping Time", "N/A")}</p>
            <p><strong>🔁 Lit-up Time:</strong> {log_data.get("Lit-up Time", "NIL")}</p>
            <p><strong>🔁 Synchronization:</strong> {log_data.get("Synchronization Time", "NIL")}</p>
            <p><strong>📌 Reason:</strong> <span style="color:#e63946;"><b>{log_data.get("Reason", "N/A")}</b></span></p>
        </div>
        """, unsafe_allow_html=True
    )

    if success:
        st.markdown(
            f"""
                    <div style="
                        background-color: rgba(255, 255, 255, 0.88);
                        color: black;
                        padding: 0.75rem 1.2rem;
                        border-radius: 10px;
                        display: inline-block;
                        font-weight: 600;
                        font-size: 16px;
                        margin-bottom: 1rem;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
                    ">
                        ✅ Reason retrieved successfully
                    </div>
                    """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div style="
                background-color: rgba(255, 255, 255, 0.8);
                color: red;
                padding: 0.75rem 1.2rem;
                border-radius: 10px;
                display: inline-block;
                font-weight: 600;
                font-size: 16px;
                margin-bottom: 1rem;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            ">
                ⚠️ No match in database.Showing ML-based prediction.
            </div>
            """,
            unsafe_allow_html=True
        )


    if st.button("⬅️ Back"):
        st.session_state.page = 1
        st.rerun()