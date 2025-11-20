import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from tqdm import tqdm

def analyze_npz(npz_path):
    """
    Analyze the distribution of data inside an npz file and save plots.
    """
    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        return

    print(f"Loading {npz_path}...")
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading npz file: {e}")
        return

    # Determine output directory
    # npz_root_dir/distribution_plots/npz_filename_folder/
    base_dir = os.path.dirname(os.path.abspath(npz_path))
    filename = os.path.splitext(os.path.basename(npz_path))[0]
    output_dir = os.path.join(base_dir, "distribution_plots", filename)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to {output_dir}")

    keys = list(data.keys())
    print(f"Found keys: {keys}")

    # Set style
    sns.set_theme(style="whitegrid")

    for key in keys:
        arr = data[key]
        print(f"Processing '{key}', shape: {arr.shape}, dtype: {arr.dtype}")

        # Handle 'y' or label-like arrays (1D integers)
        if key == 'y' or (arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer)):
            plt.figure(figsize=(12, 6))
            
            # If too many unique values, use histogram, else countplot
            unique_vals = np.unique(arr)
            if len(unique_vals) > 100:
                sns.histplot(arr, bins=50, kde=False)
                plt.title(f"Distribution of {key} (Histogram)")
            else:
                sns.countplot(x=arr)
                plt.title(f"Distribution of {key} (Count)")
                if len(unique_vals) > 20:
                    plt.xticks(rotation=90)
            
            plt.xlabel(key)
            plt.ylabel("Count")
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{key}_distribution.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {save_path}")

        # Handle 'X' or feature-like arrays
        elif key == 'X' or arr.ndim > 1:
            # 1. Overall distribution
            plt.figure(figsize=(10, 6))
            # Sample if too large for plotting
            if arr.size > 100000:
                sample_data = np.random.choice(arr.flatten(), 100000, replace=False)
                sns.histplot(sample_data, bins=100, kde=True)
                plt.title(f"Overall Value Distribution of {key} (Sampled 100k)")
            else:
                sns.histplot(arr.flatten(), bins=100, kde=True)
                plt.title(f"Overall Value Distribution of {key}")
            
            plt.xlabel("Value")
            plt.ylabel("Density/Count")
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{key}_overall_distribution.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {save_path}")

            # 2. If specific shape [N, 128, 3, 2], flatten last two dims to 6 attributes
            if arr.ndim == 4 and arr.shape[2] == 3 and arr.shape[3] == 2:
                print(f"Detected shape {arr.shape}, plotting separate features for last two dims (3x2=6 attributes)...")
                
                # Attributes: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
                for i in range(3):
                    for j in range(2):
                        feat_data = arr[:, :, i, j].flatten()
                        
                        plt.figure(figsize=(10, 6))
                        if feat_data.size > 100000:
                            sample_data = np.random.choice(feat_data, 100000, replace=False)
                            sns.histplot(sample_data, bins=100, kde=True)
                            plt.title(f"Distribution of {key} Attribute ({i}, {j}) (Sampled 100k)")
                        else:
                            sns.histplot(feat_data, bins=100, kde=True)
                            plt.title(f"Distribution of {key} Attribute ({i}, {j})")
                        
                        plt.xlabel("Value")
                        plt.ylabel("Count")
                        plt.tight_layout()
                        save_path = os.path.join(output_dir, f"{key}_attr_{i}_{j}_distribution.png")
                        plt.savefig(save_path)
                        plt.close()
                        print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NPZ dataset distribution")
    parser.add_argument("--file", type=str, default="data/train.npz", help="Path to the .npz file")
    args = parser.parse_args()

    analyze_npz(args.file)
