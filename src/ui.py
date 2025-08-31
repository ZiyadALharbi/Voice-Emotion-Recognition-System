# call libraries 
from main import analyze_emotion
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import torch
import torchaudio
import os



#title
st.set_page_config(page_title="Emotion from Voice", page_icon="🎵", layout="centered")

# Sidebar (Team Members)
st.sidebar.title("Emotion from Voice")
st.sidebar.info(
    """
    **Team Members:**  
    - XXXXXX  
    - YYYYYY 
    - ZZZZZZ  
    """
)

# Main

st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 38px !important;
            color: #4B9CD3;
        }
        .subtitle {
            text-align: center;
            font-size: 20px !important;
            color: gray;
        }
           /* Shrink uploader button */
    div.stFileUploader > label > div {
        max-width: 200px;
    }

    /* Shrink record button */
    button[title="Start Recording"], button[title="Stop Recording"] {
        padding: 0.25em 0.5em;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<p class="title"> Voice Emotion Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your voice and let our model detect the emotion</p>', unsafe_allow_html=True)



# Description
with st.expander(" About the Model"):
    st.write(
        """
        This ML model listens to a short **audio recording/ audio file**  
        and classifies the emotional tone into categories:  
        Happy   Sad   Angry 
        """
    )

st.markdown("Upload a `.wav` file **or** record your voice:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("", type=["wav"])

with col2:
    st.subheader(" Record Voice (max 10s)")
    audio = audio_recorder()


temp_file = None
# Handle uploaded file
if uploaded_file is not None:
    temp_file = "temp_uploaded.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format="audio/wav")
    
#Handle recorded file
elif audio is not None and len(audio) > 0:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(temp_file, "wb") as f:
        f.write(audio)
    st.audio(temp_file, format="audio/wav")
        
# voice player
    st.audio(uploaded_file, format="audio/wav")
    

#  button
if temp_file and st.button("Know Voice Emotion"):
    st.write("Processing...")
    """
    TODO: Replace "Processing..." with actual emotion analysis
    Add the complete processing pipeline here:

    if temp_file and st.button("Know Voice Emotion"):
        with st.spinner("Analyzing emotion..."):
            try:
                # Call the main processing function
                result = analyze_emotion(temp_file)
                
                if result['success']:
                    # Display main result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Emotion: {result['emotion'].upper()}**")
                        st.info(f"**Confidence: {result['confidence']:.2%}**")
                    with col2:
                        st.markdown(f"<h1 style='text-align: center;'>{result['emoji']}</h1>",unsafe_allow_html=True)
                else:
                    st.error(f"Error: {result['error']}")
                    
            except Exception as e:
                st.error(f"Processing failed: {e}")
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    """



"""
 TODO: Add file cleanup at the end
 Make sure to delete temporary files:
 if temp_file and os.path.exists(temp_file):
     os.unlink(temp_file)
"""