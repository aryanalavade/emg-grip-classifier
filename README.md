# EMG Grip Classifier

A machine learning pipeline for classifying hand grip types from **EMG signals recorded via two surface electrodes** (Arduino analog inputs).  
Includes data collection, feature extraction, model training, and real-time prediction from a serial stream.

---

## Project Pipeline

1. **Collect** EMG samples from Arduino → CSV  
2. **Train** a classifier on labeled grip trials using windowed time-domain features  
3. **Predict** grip type live from incoming EMG windows  

---

## Repository Structure

```

├── arduino/
│   └── emg_stream.ino
├── src/
│   ├── collect_emg.py
│   ├── train_model.py
│   └── live_predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup
Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
```

Install dependencies:
```
pip install -r requirements.txt
```

## Collect Training Data

Upload arduino/emg_stream.ino to your Arduino, then run:
```
python src/collect_emg.py --port COM3 --label power --out data_power.csv
```

Repeat for each grip label (e.g., lateral, spherical, pinch, cylindrical, hook).

## Train the Model

Place your labeled CSV files into a directory (e.g., data_labeled/) and run:
```
python src/train_model.py --data_dir data_labeled --out_model models/grip_classifier.pkl
```

The script outputs:
- classification report
- confusion matrix
- trained model (.pkl)

## Live Prediction

Run real-time grip classification from the Arduino serial stream:
```
python src/live_predict.py --port COM3 --model models/grip_classifier.pkl
```

## Notes
- Serial port names differ by OS (COM3 vs /dev/tty.usbmodemXXXX)
- Raw EMG datasets and trained models are excluded from version control by default
- Feature extraction uses time-domain EMG features (RMS, MAV, WL, ZC, SSC, variance, skewness)

## Author

**Arya Nalavade**
**M.S. Bioinformatics, Georgia Institute of Technology**
