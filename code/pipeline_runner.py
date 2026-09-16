import subprocess
import time

while True:
    print("Running Stage Data preparation...")
    subprocess.run(["python", "code/datasets/prepare_data.py"])

    print("Running Stage Model training...")
    subprocess.run(["python", "code/models/train_model.py"])

    print("Pipeline finished. Rerun in 5 minutes...")
    time.sleep(300)