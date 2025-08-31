from feature_extraction_model import EmotionRecognitionPipeline

class EmotionRunner:
    """Main runner that orchestrates the emotion recognition flow"""

    def __init__(self):
        # Initialize the pipeline once
        self.pipeline = EmotionRecognitionPipeline()

    def process_audio(self, audio_path): # Ziyad: add type
        """
        Main processing function called by UI
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Emotion recognition results
        """

        try:
            # Step 1: Extract features and predict using the pipeline
            result = self.pipeline.predict_emotion(audio_path)

            # Step 2: Format results for UI
            formatted_result = self._format_results(result)

            return formatted_result

        except Exception as e:
            return {
                # Ziyad: add it
            }
        


    def _format_results(self,result) -> dict:
        """Format results for UI consumption"""
        return {
            'success': True,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'emoji': self._get_emoji(result['emotion'])
        }

    def _get_emoji(self, emotion):
        """Get emoji for emotion"""
        emoji_map = {
            'happy': '😊', 'sad': '😢', 'angry': '😠', 
            'neutral': '😐', 'fear': '😨', 'disgust': '🤢'
        }
        return emoji_map.get(emotion, '🎭')

emotion_runner = EmotionRunner()

def analyze_emotion(audio_path):
    """Main function called by UI"""
    return emotion_runner.process_audio(audio_path)