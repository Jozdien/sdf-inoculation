"""Keep Tinker training sessions alive so the OAI proxy can serve checkpoints."""

import json
import signal
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import tinker

STATE_PATHS = {
    "run06": "tinker://d99059c8-f8f8-564f-af2d-44f80dca7379:train:0/weights/final",
    "run07": "tinker://3bf767dc-f5a9-5304-88ca-8ad923732f53:train:0/weights/final",
    "run08": "tinker://a0722f57-70a7-5dfb-bc87-b3e5d4deed84:train:0/weights/final",
    "run09": "tinker://36a895fb-f3bf-5d48-a496-cb38484eb719:train:0/weights/final",
    "run10": "tinker://1f015bab-0373-5427-98d2-81a896dac0c7:train:0/weights/final",
}


def main():
    sc = tinker.ServiceClient()
    clients = {}
    sampler_paths = {}

    for run, state_path in STATE_PATHS.items():
        print(f"Creating training client for {run}...")
        tc = sc.create_lora_training_client(
            base_model="meta-llama/Llama-3.3-70B-Instruct", rank=64
        )
        tc.load_state(state_path).result()
        sw_resp = tc.save_weights_for_sampler(f"sdf_no_hack_{run}_live").result()
        sampler_paths[run] = sw_resp.path
        clients[run] = tc
        print(f"  {run}: {sw_resp.path}")

    out_path = "/tmp/sdf_no_hack_live_paths.json"
    with open(out_path, "w") as f:
        json.dump(sampler_paths, f, indent=2)
    print(f"\nSaved paths to {out_path}")
    print("Sessions alive. Press Ctrl+C to stop.")
    sys.stdout.flush()

    def handle_signal(signum, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
