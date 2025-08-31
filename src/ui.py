import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile, os
from main import analyze_emotion

st.set_page_config(page_title="Voice Emotion", page_icon="", layout="centered")


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
      background:
        radial-gradient(900px 600px at 72% 32%, rgba(255, 105, 180, 0.10), transparent 60%),
        radial-gradient(800px 520px at 18% 28%, rgba(0, 170, 255, 0.12), transparent 55%),
        linear-gradient(135deg, #0e1224 0%, #141a36 45%, #1a1a40 70%, #22183a 100%) !important;

      background-attachment: fixed !important;
      color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



if "page" not in st.session_state:
    st.session_state.page = "home"


# Home page
def render_home():

    st.markdown(
        """
        <style>
        /* Background*/
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 25% 25%, #0d1224, #131b3a, #0b0f1a) !important;
            color: white !important;
        }

        
        .hero-wrap {
            max-width: 900px;
            margin: 70px auto 10px auto;
            padding: 0 24px;
        }

        /* Title*/
        .hero-title {
            font-size: 68px;
            line-height: 1.06;
            font-weight: 800;
            letter-spacing: -0.5px;
            text-align: left;
            color: #e8ecff;
            margin: 0 0 18px 0;
        }
        .hero-title .your {
            background: linear-gradient(90deg,#00e5ff, #00bfff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-title .voice {
            background: linear-gradient(90deg,#9d4cff, #ff2aa1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Description*/
        .subtitle {
            font-size: 18px;
            color: #c9d2ff;
            text-align: left;
            margin: 8px 0 26px 0;
            max-width: 640px;
        }

        /* Buttons*/
        .stButton > button {
            padding: 12px 26px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 15px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.06);
            color: #ffffff;
            opacity: 0.5;
            transition: transform .12s ease, opacity .7s ease, box-shadow .12s ease;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }
        .stButton > button:hover { transform: translateY(-1px); opacity: .5; }

        /*  (Get started)  */
        .stButton button:first-of-type {
            background: linear-gradient(90deg, #00c6ff, #8a3cff, #ff2aa1) !important;
            border: none !important;
            color: white !important;
        }

        
        .wave-wrap {
            position: fixed;
            right: 4%;
            bottom: 8%;
            display: flex;
            gap: 6px;
            align-items: flex-end;
        }
        .wave {
            width: 30px;
            background: linear-gradient(180deg, #ff2aa1, #7a5cff, #00c6ff);
            border-radius: 20px;
            animation: sound 1.1s infinite ease-in-out;
        }
        .wave:nth-child(1){ height: 30px; animation-delay: 0s; }
        .wave:nth-child(2){ height: 50px; animation-delay: 0.2s; }
        .wave:nth-child(3){ height: 70px; animation-delay: 0.4s; }
        .wave:nth-child(4){ height: 50px; animation-delay: 0.6s; }
        .wave:nth-child(5){ height: 30px; animation-delay: 0.8s; }

        @keyframes sound {
          0%   { transform: scaleY(0.4); }
          50%  { transform: scaleY(1.2); }
          100% { transform: scaleY(0.4); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
    st.markdown(
        """
        <div class="hero-wrap">
            <h1 class="hero-title">
                Detect emotions from <span class="your">your</span> <span class="voice">voice</span> in seconds
            </h1>
            <p class="subtitle">
                Upload or record a short clip . Our model classifies it into 
                <b>happy</b>, <b>sad</b>, <b>angry</b>, <b>neutral</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Buttons in a row
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Get started", use_container_width=True, key="start"):
            st.session_state.page = "analyze"
    with col2:
        if st.button("Learn more", use_container_width=True, key="learn"):
            st.session_state.page = "learn"
    with col3:
        if st.button("About us", use_container_width=True, key="about"):
            st.session_state.page = "about"

    
    st.markdown(
        """
        <div class="wave-wrap">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================
#Analyze page
# ======================
def render_analyze():
    import tempfile, os

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
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        f = st.file_uploader(" ", type=["wav"])
        if f:
            temp_file = "temp_uploaded.wav"
            with open(temp_file, "wb") as out: out.write(f.getbuffer())
            st.audio(temp_file, format="audio/wav")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
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
            st.warning("حمّل ملف أو سجّل مقطع أولاً.")
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
    if st.button("⬅ Back to Home"): st.session_state.page = "home"




## ======================
# Learn More page
# ======================
def render_learn():
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

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"



# ======================
# About Us page
# ======================
def render_about():
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

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"


# ======================
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
