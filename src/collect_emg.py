import argparse
import csv
import time
import serial


def parse_args():
    p = argparse.ArgumentParser(description="Collect EMG data from Arduino over serial and save to CSV.")
    p.add_argument("--port", required=True, help="Serial port (e.g., COM3 or /dev/tty.usbmodemXXXX)")
    p.add_argument("--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    p.add_argument("--out", default="emg_data.csv", help="Output CSV file")
    p.add_argument("--label", required=True, help="Grip label for this recording session (e.g., power)")
    return p.parse_args()


def main():
    args = parse_args()
    ser = serial.Serial(args.port, args.baud)
    print(f"Connected to {args.port} @ {args.baud} baud")
    time.sleep(2)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Sensor1", "Sensor2", "Label"])
        print("Logging... Press CTRL+C to stop")

        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                try:
                    s1 = int(parts[0])
                    s2 = int(parts[1])
                except ValueError:
                    continue

                writer.writerow([time.time(), s1, s2, args.label])
        except KeyboardInterrupt:
            print("Logging stopped.")

    ser.close()


if __name__ == "__main__":
    main()
