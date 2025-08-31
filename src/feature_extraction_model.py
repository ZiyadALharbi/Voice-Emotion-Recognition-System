import json, torch, torchaudio
import torchaudio.transforms as T, torchaudio.functional as AF
import torch.nn.functional as Fnn
import numpy as np
from pathlib import Path
import joblib
import xgboost as xgb


class EmotionRecognitionPipeline:
    """Handles feature extraction and model prediction"""

    def __init__(self, config_path="models/config/config.json", model_dir="models/"):
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


        # Load your config.json
        with open(config_path) as f:
            self.config = json.load(f)

        # load pre-trained XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(f"{model_dir}/xgb_gpu_hist_1.json")

        # force CPU inference
        self.model.set_param({"predictor":"cpu_predictor"})

        # load feature scaler
        self.scaler = joblib.load(f"{model_dir}/feature_scaler.joblib")

        # load model metadata
        with open(f"{model_dir}/xgb_gpu_hist_1.meta.json") as f:
            meta = json.load(f)
            self.emotion_labels = meta["classes"]
            self.n_classes = meta["n_classes"]
            self.feature_dim = meta["feature_dim"]

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
        """Extract the features"""
        features = self.feature_extractor.from_path(audio_path)

        print(f"Extracted features shape: {features.shape}")
        print(f"Features mean: {np.mean(features):.4f}, std: {np.std(features):.4f}")
        print(f"Features min: {np.min(features):.4f}, max: {np.max(features):.4f}")

        return features


    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Apply preprocessing (Scaling) - same as training"""
        # Reshape to 2D if needed (beacuse scaler expects 2D) 
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features using the same scaler from training
        features_scaled = self.scaler.transform(features)

        print(f"After scaling - mean: {np.mean(features_scaled):.4f}, std: {np.std(features_scaled):.4f}")
        
        return features_scaled.astype(np.float32)


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

        # Create DMatrix for XGBoost
        dmatrix = xgb.DMatrix(features)
        
        # Get prediction probabilities
        probabilities = self.model.predict(dmatrix)[0]  
        
        # Get predicted class
        prediction = np.argmax(probabilities)
        
        # Format results
        emotion = self.emotion_labels[prediction]
        confidence = float(probabilities[prediction])
        
        prob_dict = {
            self.emotion_labels[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': prob_dict
        }


class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg

        self.SR = cfg["sr"]
        self.N_FFT = cfg["n_fft"]
        self.WIN = int(self.SR * cfg["win_ms"] / 1000)
        self.HOP = int(self.SR * cfg["hop_ms"] / 1000)
        self._EPS = 1e-10

        self.mel  = T.MelSpectrogram(
            sample_rate=self.SR, n_fft=self.N_FFT, hop_length=self.HOP, win_length=self.WIN,
            n_mels=cfg["n_mels"], power=2.0
        )
        self.mfcc = T.MFCC(
            sample_rate=self.SR, n_mfcc=cfg["n_mfcc"],
            melkwargs={"n_fft":self.N_FFT, "hop_length":self.HOP, "win_length":self.WIN, "n_mels":cfg["n_mels"]}
        )
        self.spec = T.Spectrogram(n_fft=self.N_FFT, hop_length=self.HOP, win_length=self.WIN, power=2.0)

    @torch.no_grad()
    def from_path(self, path:str) -> np.ndarray:
        SR = self.SR
        _EPS = self._EPS

        # load & standardize
        y, sr0 = torchaudio.load(path)              # [C, T]
        print(f"Loaded audio: shape={y.shape}, sr={sr0}, duration={y.shape[1]/sr0:.2f}s")

        y = y.mean(0, keepdim=True)
        if sr0 != SR:
            y = torchaudio.functional.resample(y, sr0, SR)
        peak = float(y.abs().max())
        if peak > 0:
            y = y * (self.cfg["peak_target"] / peak)

        # frame features
        mel = self.mel(y).clamp_min(_EPS)                 # [1, M, F]
        logmel = torch.log(mel).squeeze(0).T              # [F, M]

        mfcc = self.mfcc(y).squeeze(0).T                  # [F, C]
        d1   = AF.compute_deltas(mfcc.T).T
        d2   = AF.compute_deltas(d1.T).T

        spec = self.spec(y).squeeze(0).clamp_min(_EPS)    # [K, F]
        F_frames = spec.shape[1]
        freqs = torch.linspace(0, SR/2, spec.shape[0], device=spec.device)
        ps = spec                                         # already >= eps

        # spectral shape
        cen = (freqs[:,None] * ps).sum(0) / ps.sum(0)
        bw  = torch.sqrt(((freqs[:,None] - cen[None,:])**2 * ps).sum(0) / ps.sum(0))

        # rolloffs (contiguous for searchsorted)
        cs = torch.cumsum(ps, dim=0).contiguous()
        tot = cs[-1,:].contiguous()
        t85 = (0.85*tot).unsqueeze(1).contiguous()
        t95 = (0.95*tot).unsqueeze(1).contiguous()
        idx85 = torch.searchsorted(cs.T.contiguous(), t85).clamp(max=cs.shape[0]-1).squeeze(1)
        idx95 = torch.searchsorted(cs.T.contiguous(), t95).clamp(max=cs.shape[0]-1).squeeze(1)
        roll85, roll95 = freqs[idx85], freqs[idx95]

        # spectral flatness (geom/arith mean) — numerically safe
        geo = torch.exp(torch.log(ps).mean(0))
        arith = ps.mean(0).clamp_min(_EPS)
        flat = (geo / arith)

        # spectral flux (on L2-normalized magnitude)
        mag = torch.sqrt(ps)
        mag = mag / (mag.norm(p=2, dim=0, keepdim=True).clamp_min(_EPS))
        flux = torch.zeros(mag.shape[1], device=mag.device)
        flux[1:] = (mag[:,1:] - mag[:,:-1]).pow(2).sum(0).sqrt()

        # frame energy in dB (finite by construction)
        frame_energy_db = 10.0 * torch.log10(ps.mean(0).clamp_min(_EPS))

        # pitch (no hop_length/win_length) + align to spectrogram frames
        f0_raw = AF.detect_pitch_frequency(
            y, sample_rate=SR, frame_time=self.cfg["win_ms"]/1000.0
        ).squeeze(0)                              # [F0_frames]
        if f0_raw.numel() == 0:
            f0_rs = torch.zeros(F_frames, device=spec.device)
        else:
            f0_in = f0_raw.clone()
            f0_in[f0_in <= 0] = 0.0              # unvoiced -> 0 for interpolation
            f0_rs = Fnn.interpolate(
                f0_in.view(1,1,-1), size=F_frames, mode="linear", align_corners=False
            ).view(-1)
        voiced = f0_rs > 0
        f0 = torch.where(voiced, f0_rs, torch.nan)       # keep NaN for unvoiced; pooling handles it

        # stack frames x dims (KEEPING ALL FEATURES)
        F = torch.cat([
            mfcc, d1, d2,
            logmel,
            torch.stack([cen, bw, roll85, roll95, flat, flux, frame_energy_db], dim=1),
            f0.unsqueeze(1),
        ], dim=1)


        F = torch.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

        # pooling from config (all + voiced-only; fixed size even if no voiced)
        def pool(A: torch.Tensor) -> torch.Tensor:
            parts = []
            if "mean"   in self.cfg["pooling"]: parts.append(A.mean(0))
            if "std"    in self.cfg["pooling"]: parts.append(A.std(0))
            if "median" in self.cfg["pooling"]: parts.append(A.median(0).values)
            if "p10"    in self.cfg["pooling"]: parts.append(torch.quantile(A, 0.10, dim=0))
            if "p90"    in self.cfg["pooling"]: parts.append(torch.quantile(A, 0.90, dim=0))
            if "slope"  in self.cfg["pooling"]:
                t = torch.linspace(0, 1, A.shape[0], device=A.device).unsqueeze(1)
                den = ((t - t.mean())**2).sum().clamp_min(1e-9)
                slope = (t * (A - A.mean(0))).sum(0) / den
                parts.append(slope)
            return torch.cat(parts, 0)

        v_all = pool(F)
        # voiced stats vector (same length as v_all); if no voiced frames, fill zeros to keep dimension
        if self.cfg.get("voiced_variant", True):
            if voiced.any():                v_vo = pool(F[voiced])
            else:
                v_vo = torch.zeros_like(v_all)
            v = torch.cat([v_all, v_vo], 0)
        else:
            v = v_all

        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v.float().cpu().numpy()
