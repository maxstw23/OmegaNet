import uproot
import awkward as ak
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_star_counts():
    file_path = "data/omega_kaon_all.root"

    with uproot.open(file_path) as file:
        tree = file["ml_tree"]

        # 1. Load data
        # run_id and evt_id are flat (one per event)
        # kaon_charge is jagged (vector per event)
        data = tree.arrays(["run_id", "evt_id", "kaon_charge"], library="ak")

        # 2. Get the counts of kaons per event
        # This is the "map" that tells us how many times to repeat the Event IDs
        counts = ak.num(data["kaon_charge"])

        # 3. Repeat Event IDs so they match the track count
        # If event 1 has 3 kaons, we turn [Run1] into [Run1, Run1, Run1]
        # Try to view the float-misinterpreted data as the original integers
        k_run_expanded = np.repeat(ak.to_numpy(data["run_id"]), counts)
        k_evt_expanded = np.repeat(ak.to_numpy(data["evt_id"]), counts)

        # Now we flatten the charges (the only truly jagged part)
        k_charge_flat = ak.to_numpy(ak.flatten(data["kaon_charge"]))

        # 4. Key Generation
        # Track-level keys (length = 4,789,939)
        k_keys = k_run_expanded.astype(np.int64) * 1_000_000_000 + k_evt_expanded.astype(np.int64)

        # Omega-level keys (length = 166,934)
        o_keys = ak.to_numpy(data["run_id"]).astype(np.int64) * 1_000_000_000 + ak.to_numpy(data["evt_id"]).astype(
            np.int64)
        unique_o_keys = np.unique(o_keys)

        # 5. Mapping
        kaon_event_idx = np.searchsorted(unique_o_keys, k_keys)

        # 6. Masking and Counting
        kp_mask = (k_charge_flat > 0)
        km_mask = (k_charge_flat < 0)

        # Only count kaons belonging to our Omega events
        valid_mask = (kaon_event_idx < len(unique_o_keys))

        kp_counts = np.bincount(kaon_event_idx[kp_mask & valid_mask], minlength=len(unique_o_keys))
        km_counts = np.bincount(kaon_event_idx[km_mask & valid_mask], minlength=len(unique_o_keys))

        # 7. Consistency Report
        print(f"\n--- STAR Real Data Report ---")
        print(f"Total Unique Events: {len(unique_o_keys)}")
        print(f"K+ Mean: {np.mean(kp_counts):.4f}")
        print(f"K- Mean: {np.mean(km_counts):.4f}")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.hist(kp_counts, bins=range(60), alpha=0.5, label='$K^+$', color='blue')
        plt.hist(km_counts, bins=range(60), alpha=0.5, label='$K^-$', color='red')
        plt.yscale('log')
        plt.xlabel("Kaon Multiplicity per Event")
        plt.ylabel("Events")
        plt.legend()
        plt.savefig("kaon_multiplicity_check.png")
        print("\nSuccess! Plot saved as 'kaon_multiplicity_check.png'")


if __name__ == "__main__":
    analyze_star_counts()