# finalize_dataset.py
import os
import pandas as pd

def finalize_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    
    input_path = os.path.join(root_dir, "dataset", "netflix_cleaned.csv")
    output_path = os.path.join(root_dir, "dataset", "final_cleaned_dataset.csv")
    
    # 1. Load cleaned dataset
    df = pd.read_csv(input_path)
    print("Initial dataset shape:", df.shape)

    # -------- EDA transformations --------

    # 2. Split multiple countries into separate rows
    if "country" in df.columns:
        df["country"] = df["country"].str.split(", ")
        df = df.explode("country")

    # 3. Convert duration into numeric (minutes / seasons as ~60 mins)
    def convert_duration(val):
        if isinstance(val, str):
            if "min" in val:
                return int(val.replace(" min", ""))
            elif "Season" in val:
                return int(val.split(" ")[0]) * 60
        return None

    df["duration_num"] = df["duration"].apply(convert_duration)

    # 4. Create binary columns for type
    df["is_movie"] = (df["type"] == "Movie").astype(int)
    df["is_show"] = (df["type"] == "TV Show").astype(int)

    # 5. Reset index for clean rows
    df.reset_index(drop=True, inplace=True)

    # 6. Save final dataset
    df.to_csv(output_path, index=False)
    print(f"Final dataset saved as: {output_path}")
    print("Final dataset shape:", df.shape)


if __name__ == "__main__":
    finalize_dataset()
