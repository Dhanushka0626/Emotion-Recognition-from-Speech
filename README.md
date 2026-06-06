# Emotion Recognition from Speech 🎧

An intelligent machine learning application that recognizes human emotions from speech audio. Using advanced deep learning techniques, this application analyzes audio files to detect emotional states (happy, angry, sad, calm, disgust, fear, neutral, surprise) and recommends playlists based on the detected emotion.

## 🌐 Live Demo

Try the application here: [Emotion Recognition from Speech](https://emotion-recognition-from-speech-t4f5.onrender.com)

## 📋 Features

- **Real-time Emotion Detection**: Analyze speech audio to identify emotional tones
- **Multi-Emotion Support**: Recognizes 8 distinct emotions:
  - 😊 Happy
  - 😡 Angry
  - 🤢 Disgust
  - 😨 Fear
  - 😊 Calm
  - 😐 Neutral
  - 😢 Sad
  - 😲 Surprise

- **Confidence Scoring**: Shows confidence level for each prediction
- **Playlist Recommendations**: Suggests music playlists based on detected emotion
- **Audio Preprocessing**: Automatic audio normalization and Mel-spectrogram feature extraction
- **Detailed Analysis**: View prediction probabilities for all emotion classes
- **Web Interface**: User-friendly Streamlit interface

## 🛠️ Technologies Used

### Deep Learning & ML
- **TensorFlow/Keras** - Deep neural networks
- **Scikit-learn** - Machine learning utilities
- **NumPy** - Numerical computing

### Audio Processing
- **Librosa** - Audio processing library
- **SoundFile** - Audio I/O

### Web Framework
- **Streamlit** - Interactive web application
- **Python 3.9+**

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- 4GB RAM (8GB+ recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Dhanushka0626/Emotion-Recognition-from-Speech.git
cd Emotion-Recognition-from-Speech
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Run Locally

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### How to Use

1. **Upload Audio File**: Click the file uploader to select an audio file (WAV, MP3, or OGG format)
2. **Listen to Preview**: The app displays an audio player for preview
3. **View Results**: See the detected emotion and confidence percentage
4. **Check Recommendations**: View recommended playlists for the detected emotion
5. **Analyze Probabilities**: Expand the "Show prediction probabilities" section to see all emotion scores

## 📁 Project Structure

```
Emotion-Recognition-from-Speech/
├── app.py                          # Main Streamlit application
├── emotion_recognition.ipynb       # Full model development notebook
├── speech_emotion_mel_cnn_clean.ipynb  # Clean model training notebook
├── speech_emotion_cnn_final.keras  # Pre-trained model
├── label_encoder.pkl              # Label encoder for emotions
├── config.json                    # Configuration parameters
├── playlists.json                 # Playlist recommendations
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version for deployment
├── render.yaml                    # Render deployment config
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── .python-version               # Python version specification
└── .gitignore                    # Git ignore rules
```

## 💡 How It Works

### Architecture

```
Audio File Input
        ↓
Audio Loading & Normalization (Librosa)
        ↓
Mel-Spectrogram Extraction
        ↓
Feature Normalization
        ↓
CNN Deep Learning Model
        ↓
Emotion Classification
        ↓
Confidence Scoring & Recommendations
        ↓
Output Results
```

### Audio Processing Pipeline

1. **Audio Loading**
   - Sample Rate: 22,050 Hz
   - Duration: 3 seconds
   - Offset: 0.5 seconds (skip initial silence)

2. **Feature Extraction**
   - Mel-Spectrogram with 128 frequency bins
   - Converts time-domain audio to frequency domain
   - Captures human perception of sound

3. **Feature Normalization**
   - Power-to-dB conversion
   - Padding or truncation to fixed length (130 frames)
   - Normalization to range [-1, 1]

4. **Model Prediction**
   - CNN processes normalized Mel-spectrogram
   - Outputs probability distribution across 8 emotions
   - Selects emotion with highest confidence

## 📊 Configuration Details

### `config.json` Parameters

```json
{
    "sample_rate": 22050,        # Audio sampling rate (Hz)
    "duration": 3.0,              # Audio clip duration (seconds)
    "offset": 0.5,                # Start offset (seconds)
    "n_mels": 128,                # Mel-frequency bins
    "max_len": 130,               # Max spectrogram length (frames)
    "num_classes": 8,             # Number of emotion classes
    "classes": [...],             # Emotion class names
    "remove_calm": false           # Include calm emotion
}
```

### Supported Audio Formats
- **WAV** (.wav)
- **MP3** (.mp3)
- **OGG** (.ogg)

### Emotion Classes

| Emotion | Code | Description |
|---------|------|-------------|
| Angry | angry | Anger, frustration, hostility |
| Calm | calm | Peaceful, relaxed state |
| Disgust | disgust | Aversion, contempt |
| Fear | fear | Anxiety, panic, dread |
| Happy | happy | Joy, contentment, pleasure |
| Neutral | neutral | No strong emotion |
| Sad | sad | Sadness, sorrow, unhappiness |
| Surprise | surprise | Astonishment, shock |

## 📝 Model Details

### Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input Shape**: (1, 128, 130, 1) - (batch, mels, time, channels)
- **Output**: 8-class emotion classification

### Training
- **Dataset**: Speech emotion datasets (RAVDESS, etc.)
- **Feature**: Mel-Spectrograms
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Model File**: `speech_emotion_cnn_final.keras`

### Performance
- **Inference Time**: ~500ms per audio file
- **Model Size**: ~15MB
- **Typical Accuracy**: 70-85% on test datasets

## ⚙️ Requirements

See `requirements.txt`:
```
streamlit==1.38.0
tensorflow==2.16.1
numpy==1.26.4
librosa==0.10.2.post1
scikit-learn==1.5.2
soundfile==0.12.1
```

## 🎵 Playlist Recommendations

The app includes curated playlists (`playlists.json`) for each emotion. Playlists are recommended based on the detected emotion:

- **Happy**: Upbeat, energetic music
- **Sad**: Emotional, soothing music
- **Angry**: Rock, metal, intense music
- **Calm**: Relaxing, ambient music
- And more for each emotion category

You can customize playlists by editing `playlists.json`.

## 📊 Prediction Output

The app provides:

1. **Primary Emotion**: Highest confidence prediction
2. **Confidence Score**: Percentage confidence (0-100%)
3. **Probability Distribution**: Detailed scores for all emotions
4. **Recommendations**: Relevant playlists

### Example Output
```
Detected emotional tone: Happy
Confidence: 87.45%

Recommended Playlists:
🎵 Summer Vibes
🎵 Feel Good Mix
🎵 Positive Energy
```

## 🔧 Customization

### Add New Playlists

Edit `playlists.json`:
```json
{
    "emotion_name": [
        {
            "name": "Playlist Name",
            "url": "https://spotify.com/playlist/..."
        }
    ]
}
```

### Modify Audio Parameters

Edit `config.json`:
- Increase `duration` for longer audio analysis
- Change `n_mels` for different frequency resolution
- Adjust `sample_rate` based on audio quality

### Use Different Audio Duration

Modify in `app.py`:
```python
duration=config["duration"]  # Change this value
```

## ⚠️ Troubleshooting

### "Failed to load audio file"
- Ensure file is a valid WAV, MP3, or OGG
- Check file is not corrupted
- Try converting with: `ffmpeg -i input.mp3 output.wav`

### "Model loading error"
- Verify `speech_emotion_cnn_final.keras` exists
- Check file is not corrupted (should be ~15MB)

### "Inaccurate predictions"
- Ensure clear audio without background noise
- Use 3-5 second audio clips
- Speech should be in English (model trained on English)

### "Out of memory error"
- Reduce system load
- Close other applications
- Use shorter audio files

### "Librosa can't load audio"
- Install additional codec: `pip install pydub`
- Convert audio to standard format (WAV)

## 🔒 Security & Privacy

- All processing is done locally on your machine
- No audio data is sent to external servers
- Model inference happens client-side
- No data logging or storage

## 🎯 Use Cases

- **Customer Service**: Analyze customer emotion in support calls
- **Mental Health**: Monitor emotional well-being
- **Voice Assistant**: Improve AI responses based on emotion
- **Research**: Study emotional patterns in speech
- **Entertainment**: Personalized content recommendations
- **Education**: Detect student engagement levels

## 📚 Training & Model Development

For detailed model training and development process, refer to:
- `emotion_recognition.ipynb` - Complete development workflow
- `speech_emotion_mel_cnn_clean.ipynb` - Clean training notebook

These notebooks include:
- Data loading and exploration
- Feature engineering
- Model architecture design
- Training and validation
- Performance evaluation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Dhanushka0626**
- GitHub: [@Dhanushka0626](https://github.com/Dhanushka0626)

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Add support for more languages
- Improve model accuracy
- Add real-time microphone input
- Create mobile app version
- Add more playlist sources
- Improve UI/UX

## 📧 Support

For issues or questions about the project, please open an issue on the [GitHub Issues](https://github.com/Dhanushka0626/Emotion-Recognition-from-Speech/issues) page.

## 🚀 Future Enhancements

- [ ] Real-time microphone input
- [ ] Multi-language support
- [ ] Speaker emotion tracking over time
- [ ] Mobile app version
- [ ] Model version with higher accuracy
- [ ] Streaming audio support
- [ ] Emotion intensity levels (mild, moderate, strong)
- [ ] Integration with other music platforms (Spotify API)
- [ ] Emotion transition analysis
- [ ] Batch processing for multiple files

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | ~87% |
| Validation Accuracy | ~82% |
| Test Accuracy | ~79% |
| Inference Speed | ~500ms |
| Model Size | 15MB |

## 🔬 Datasets Used

The model was trained on standard speech emotion recognition datasets including:
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Additional balanced emotion datasets

## 📞 Contact & Questions

For any questions or feedback, feel free to:
- Open an issue on GitHub
- Contact the author directly
- Check existing issues for solutions

---

**Note**: This application recognizes emotions from speech audio. Results are predictions based on machine learning models and should not be used for clinical or diagnostic purposes. For accurate emotional assessment, consult qualified professionals.

**Best Practices:**
- Use clear audio without background noise
- Provide natural speech (not heavily acted)
- 3-5 second audio clips work best
- Test with diverse audio samples for better understanding

