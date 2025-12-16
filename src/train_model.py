import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def extract_features(df, window_size=50, ssc_thresh=8):
    features = []
    labels = []

    for label in df["label"].unique():
        grip_df = df[df["label"] == label][["Sensor1", "Sensor2"]].values.astype(float)

        for i in range(0, len(grip_df) - window_size, window_size):
            window = grip_df[i : i + window_size]
            feature_vector = []

            for col in range(window.shape[1]):
                signal = window[:, col]
                std = np.std(signal)
                signal = (signal - np.mean(signal)) / std if std != 0 else signal - np.mean(signal)

                rms = np.sqrt(np.mean(signal**2))
                mav = np.mean(np.abs(signal))
                wl = np.sum(np.abs(np.diff(signal)))
                zc = np.sum(np.diff(np.signbit(signal)) != 0)
                diff1 = np.diff(signal[:-1])
                diff2 = np.diff(signal[1:])
                ssc = np.sum((diff1 * diff2 < 0) & (np.abs(diff1 - diff2) > ssc_thresh))

                var = np.var(signal)
                skew = np.mean((signal - np.mean(signal)) ** 3) / (np.std(signal) ** 3 + 1e-6)

                feature_vector.extend([rms, mav, wl, zc, ssc, var, skew])

            features.append(feature_vector)
            labels.append(label)

    return np.array(features), np.array(labels)


def parse_args():
    p = argparse.ArgumentParser(description="Train EMG grip classifier from labeled CSV files.")
    p.add_argument("--data_dir", required=True, help="Directory containing labeled CSV files")
    p.add_argument("--out_model", default="models/grip_classifier.pkl", help="Output model path")
    p.add_argument("--window", type=int, default=50, help="Window size for feature extraction")
    return p.parse_args()


def main():
    args = parse_args()

    csvs = [f for f in os.listdir(args.data_dir) if f.lower().endswith(".csv")]
    if not csvs:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    frames = []
    for f in csvs:
        path = os.path.join(args.data_dir, f)
        df = pd.read_csv(path)

        # Expect either an existing Label column OR infer label from filename
        if "label" not in df.columns and "Label" in df.columns:
            df["label"] = df["Label"]
        elif "label" not in df.columns:
            inferred = os.path.splitext(f)[0]
            df["label"] = inferred

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    X, y = extract_features(combined, window_size=args.window)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(clf, args.out_model)
    print(f"\nModel saved to {args.out_model}")

    # Save feature importance plot
    feature_labels = [
        "RMS_S1","MAV_S1","WL_S1","ZC_S1","SSC_S1","VAR_S1","SKEW_S1",
        "RMS_S2","MAV_S2","WL_S2","ZC_S2","SSC_S2","VAR_S2","SKEW_S2",
    ]
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(9, 6))
    plt.barh(range(len(importances)), importances[sorted_idx], align="center")
    plt.yticks(range(len(importances)), np.array(feature_labels)[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/feature_importance.png", dpi=200)
    print("Saved figures/feature_importance.png")


if __name__ == "__main__":
    main()
