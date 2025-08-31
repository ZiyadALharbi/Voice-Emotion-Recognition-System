import json, torch, torchaudio
import torchaudio.transforms as T, torchaudio.functional as AF
import torch.nn.functional as Fnn
import numpy as np
from pathlib import Path
import pickle


class EmotionRecognitionPipeline:
    """Handles feature extraction and model prediction"""

    def __init__(self, config_path="config.json", model_dir="models/"):
        """
        Flow:
        1. load configuration [DONE]
        2. Load pre-trained model componnets
        3. Initialize feature extraction  [DONE]
        4. Emotion labels 

        Examples (just examples!!!)
        -------------
        Load pre-trained model componnets:

        >>> self.model = pickle.load(open(f"{model_dir}/trained_model.pkl", "rb"))
        >>> self.scaler = pickle.load(open(f"{model_dir}/scaler.pkl", "rb"))
        >>> self.pca = pickle.load(open(f"{model_dir}/pca.pkl", "rb"))

        Emotion labels:
        >>> self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        """


        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
        
        SR = self.config["sr"]; N_FFT = self.config["n_fft"]
        WIN = int(SR*self.config["win_ms"]/1000); HOP = int(SR*self.config["hop_ms"]/1000)
        _EPS = 1e-10

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.config)


    def predict_emotion(self, audio_path: str) -> dict:
        """
        Complete pipeline: Audio → Features → Prediction
        
        Flow:
        1. Load audio file
        2. Extract features using TorchAudio
        3. Apply preprocessing (U-MAP)
        4. Predict using trained model
        5. Return formatted results
        """

        # Step 1: Extract features from audio
        features = self._extract_features(audio_path)
        
        # Step 2: Preprocess features (same as training)
        features_processed = self._preprocess_features(features)
        
        # Step 3: Make prediction
        prediction_result = self._predict(features_processed)
        
        return prediction_result

    
    def _extract_features(self, audio_path: str) -> np.ndarray:
        """Extract the feature vector"""


    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Apply preprocessing eg. U-MAP"""


    def _predict(self, features: np.ndarray) -> dict:
        """
        Make prediction and return probabilities

        Example
        -------------
        >>> # Get prediction and probabilities
        >>> prediction = self.model.predict(features)[0]
        >>> probabilities = self.model.predict_proba(features)[0]
        
        >>> # Format results
        >>> emotion = self.emotion_labels[prediction]
        >>> confidence = float(max(probabilities))
        
        >>> prob_dict = {
        >>>    self.emotion_labels[i]: float(prob) 
        >>>    for i, prob in enumerate(probabilities)
        >>> }
        
        >>> return {
        >>>    'emotion': emotion,
        >>>    'confidence': confidence,
        >>>    'probabilities': prob_dict
        >>> }
        """



class FeatureExtractor:
    """Add the existing feature extraction class from notebook"""

    def __init__(self, cfg):
        # Paste the code HERE!
        pass


    def from_path(self, path:str) -> np.ndarray:
        # Paste the code HERE!
        pass