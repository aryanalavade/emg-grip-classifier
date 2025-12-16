import argparse
import time
import numpy as np
import joblib
import serial


def parse_args():
    p = argparse.ArgumentParser(description="Live EMG grip prediction from Arduino serial stream.")
    p.add_argument("--port", required=True, help="Serial port (e.g., COM3 or /dev/tty.usbmodemXXXX)")
    p.add_argument("--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    p.add_argument("--model", default="models/grip_classifier.pkl", help="Path to trained model .pkl")
    p.add_argument("--window", type=int, default=50, help="Samples per prediction window")
    return p.parse_args()


def window_features(buffer1, buffer2, ssc_thresh=5):
    feats = []
    for signal in [buffer1, buffer2]:
        signal = np.array(signal, dtype=float)
        std = np.std(signal)
        signal = (signal - np.mean(signal)) / std if std != 0 else signal - np.mean(signal)

        rms = np.sqrt(np.mean(signal**2))
        mav = np.mean(np.abs(signal))
        wl = np.sum(np.abs(np.diff(signal)))
        zc = np.sum(np.diff(np.signbit(signal)) != 0)
        diff1 = np.diff(signal[:-1])
        diff2 = np.diff(signal[1:])
        ssc = np.sum((diff1 * diff2 < 0) & (np.abs(diff1 - diff2) > ssc_thresh))

        feats.extend([rms, mav, wl, zc, ssc])
    return np.array(feats).reshape(1, -1)


def main():
    args = parse_args()
    clf = joblib.load(args.model)

    ser = serial.Serial(args.port, args.baud)
    time.sleep(2)

    buffer1, buffer2 = [], []
    print("Listening for EMG... Press Ctrl+C to stop.\n")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            parts = line.split(",")
            if len(parts) != 2:
                continue

            try:
                emg1 = float(parts[0])
                emg2 = float(parts[1])
            except ValueError:
                continue

            buffer1.append(emg1)
            buffer2.append(emg2)

            if len(buffer1) >= args.window:
                feats = window_features(buffer1[:args.window], buffer2[:args.window])
                pred = clf.predict(feats)[0]
                print(f"Prediction: {pred}")

                buffer1.clear()
                buffer2.clear()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
