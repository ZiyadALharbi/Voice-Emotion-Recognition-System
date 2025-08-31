import base64
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile, os
from main import analyze_emotion 


st.set_page_config(page_title="Voice Emotion", page_icon="", layout="wide")

# session_state
if "page" not in st.session_state:
    st.session_state.page = "home"

IMG_PATH = "waves.jpeg"   
def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_b64 = b64(IMG_PATH)

st.markdown(f"""
<style>
/* صورة الخلفية كعنصر ثابت */
#_app_bg {{
position: fixed;
inset: 0;
width: 100vw; height: 100vh;
object-fit: cover;
  z-index: -1;               
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
background: transparent !important;
}}
[data-testid="stHeader"], footer, #MainMenu {{ background: transparent !important; }}
</style>
<img id="_app_bg" src="data:image/jpeg;base64,{img_b64}" />
""", unsafe_allow_html=True)

# ====================== Home ======================
def render_home():
    # ===== Navbar =====
    st.markdown(
        """
        <style>
        .navbar {
            position: absolute;
            top: 20px; 
            left: 20px;
            display: flex;
            gap: 12px;
        }
        .stButton > button {
            padding: 6px 18px;
            border-radius: 8px;
            background: rgba(255,255,255,0.05);
            border: none !important;   /* يشيل الحواف */
            color: white;
            font-weight: 500;
            cursor: pointer;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg,#00c6ff,#7a5cff,#ff2aa1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([0.12, 0.88])
    with col1:
        if st.button("About us", key="about_nav_top"):   
            st.session_state.page = "about"
    with col2:
        if st.button("Learn more", key="learn_nav_top"):  
            st.session_state.page = "learn"

    # ===== Title=====
    st.markdown(
    """
    <div style="margin-top:70px;">
        <h1 style='text-align:left; font-size:25x; font-weight:650;'>
            Detect 
            <span style="background: linear-gradient(90deg,#00c6ff,#7a5cff,#ff2aa1);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;">
                voice
            </span> 
            <span style="background: linear-gradient(90deg,#ff2aa1,#ff5cff,#ffa3ff);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;">
                Emotions
            </span> 
            in seconds
        </h1>
        <p style='text-align:left; margin-bottom:20px; color:#c9d2ff; font-size:17px;'>
            Upload or record a short clip. Our model classifies your emotions instantly.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown(
        """
        </h1>
        
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Get started", key="start"):
        st.session_state.page = "analyze"

    # ====== About us + Learn more ======
    st.markdown("""
    <div class="wave-wrap">
        <div class="wave"></div><div class="wave"></div><div class="wave"></div>
        <div class="wave"></div><div class="wave"></div>
    </div>
    """, unsafe_allow_html=True)


#Analyze page
# ======================
def render_analyze():
    st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, 
            #050507 10%,     
            #0d0820 60%,    
            #1a1033 70%,    
            #060a18 100%    
        ) !important;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    import tempfile, os
    st.markdown("""
<style>
#_app_bg { display:none !important; }

.stApp {
  background-color: #000000 !important; 
  color: #ffffff !important;           
}
</style>
""", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        :root{
          --fg:#e8ecff; --muted:#c9d2ff;
          --glass-bg:rgba(255,255,255,.06); --glass-bd:rgba(255,255,255,.18);
        }
        .h1{font:800 32px/1.2 system-ui; color:var(--fg); margin:6px 0 4px}
        .p {color:var(--muted); margin:0 0 18px}
        .glass{background:var(--glass-bg); border:1px solid var(--glass-bd);
               border-radius:12px; padding:14px; box-shadow:0 8px 24px rgba(0,0,0,.25)}
        .btn-grad .stButton>button{
          width:100%; padding:12px 22px; border-radius:12px; font-weight:800;
          background:linear-gradient(90deg,#00c6ff,#8a3cff,#ff2aa1)!important;
          border:none!important; color:#fff!important; box-shadow:0 10px 26px rgba(0,0,0,.28)
        }
        .res{margin-top:14px} .res h3{margin:0; color:var(--fg)} .res small{color:var(--muted)}
        .emoji{font-size:40px; margin-right:10px}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='h1'>Analyze Emotions in Seconds</div>", unsafe_allow_html=True)
    st.markdown("<div class='p'>Upload or record a short clip (10s), then click Analyze.</div>", unsafe_allow_html=True)

    temp_file = None
    tab1, tab2 = st.tabs(["Upload (.wav)", "Record (≤10s)"])

    with tab1:
        f = st.file_uploader(" ", type=["wav"])
        if f:
            temp_file = "temp_uploaded.wav"
            with open(temp_file, "wb") as out: out.write(f.getbuffer())
            st.audio(temp_file, format="audio/wav")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        audio_bytes = audio_recorder()
        if audio_bytes:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tf.write(audio_bytes); tf.flush()
            temp_file = tf.name
            st.audio(temp_file, format="audio/wav")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='btn-grad'>", unsafe_allow_html=True)
    go = st.button("Analyze emotion")
    st.markdown("</div>", unsafe_allow_html=True)

    if go:
        if not temp_file:
            st.warning("Download or record a clip first")
        else:
            with st.spinner("Analyzing..."):
                
                # from inference import analyze_emotion
                result = analyze_emotion(temp_file)
                # result = {"success": True, "emotion": "happy", "confidence": 0.92, "emoji": "😄"}
            if result.get("success"):
                st.markdown(
                    f"<div class='glass res'><span class='emoji'>{result['emoji']}</span>"
                    f"<h3>Result: {result['emotion'].upper()}</h3>"
                    f"<small>Confidence: {result['confidence']:.1%}</small></div>",
                    unsafe_allow_html=True
                )

                # Show detailed probabilities
                if result.get('probabilities'):
                    st.markdown("### Detailed Probabilities:")
                    for emotion, prob in result['probabilities'].items():
                        st.write(f"**{emotion.capitalize()}**: {prob:.1%}")
            else:
                st.error(result.get("error","Unknown error"))
            try:
                if temp_file and os.path.exists(temp_file): os.unlink(temp_file)
            except: pass

    st.markdown("---")
    if st.button("Back"): st.session_state.page = "home"




# Learn More page
# ======================
def render_learn():
    st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, 
            #050507 10%,     
            #0d0820 60%,    
            #1a1033 70%,    
            #060a18 100%    
        ) !important;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown("""
<style>
#_app_bg { display:none !important; }

.stApp {
  background-color: #000000 !important; /* أسود سادة */
  color: #ffffff !important;           /* نصوص بيضاء */
}
</style>
""", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .learn-title {
            font-size: 42px; font-weight: 800; margin-bottom: 16px;
            background: linear-gradient(90deg,#00c6ff,#8a3cff,#ff2aa1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .learn-text {
            font-size: 18px; color: #c9d2ff; line-height: 1.6;
            max-width: 720px; margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='learn-title'> Learn More</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p class='learn-text'>
        Our project <b>Voice Emotion Recognition</b> is designed to detect emotions 
        directly from your voice in just a few seconds.  
        The model analyzes short audio clips  and classifies.
        
        </p>

        <p class='learn-text'>
          <b>How it works:</b><br>
        1. Upload or record a short voice clip.<br>
        2. The audio is preprocessed using <b>signal processing</b> & <b>ML techniques</b>.<br>
        3. The model predicts the emotion and shows the confidence level.
        </p>

        <p class='learn-text'>
        This project can be used in <b>customer service, call centers, education, 
        and AI assistants</b> to better understand emotions 
        and improve communication.
        </p>
        """,
        unsafe_allow_html=True
    )

    if st.button("Back"):
        st.session_state.page = "home"

# About Us page
# ======================
def render_about():
    st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, 
            #050507 10%,     /* أسود أعمق */
            #0d0820 60%,    /* بنفسجي غامق جدًا */
            #1a1033 70%,    /* بنفسجي مزرق داكن */
            #060a18 100%    /* أزرق كحلي غامق */
        ) !important;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown("""
<style>
#_app_bg { display:none !important; }

.stApp {
  background-color: #000000 !important; /* أسود سادة */
  color: #ffffff !important;           /* نصوص بيضاء */
}
</style>
""", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .team-title {
            font-size: 32px; font-weight: 800; margin-bottom: 20px; color: #e8ecff;
        }
        .team-list {
            list-style: none;
            padding-left: 0;
        }
        .team-list li {
            font-size: 20px;
            margin: 8px 0;
            padding: 10px 16px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            color: #ffffff;
            transition: transform 0.2s ease;
        }
        .team-list li:hover {
            transform: translateX(6px);
            background: linear-gradient(90deg,#00c6ff,#7a5cff,#ff2aa1);
            color: white;
        }
        </style>

        <h2 class="team-title"> Our Team</h2>
        <ul class="team-list">
            <li>Azzam Al-Jariwi</li>
            <li>Raghad Al-Ghamdi</li>
            <li>Ruwaa Surrati</li>
            <li>Maymoonah Alolah</li>
            <li>Ziyad Al-Harbi</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    if st.button(" Back"):
        st.session_state.page = "home"

# Guidance
# ======================
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "analyze":
    render_analyze()
elif st.session_state.page == "learn":
    render_learn()
elif st.session_state.page == "about":
    render_about()
