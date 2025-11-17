import os
import openml
import pandas as pd


OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    "adult": 1590,
    "numerai": 23517,
    "magic_telescope": 1120,
    "kr_vs_kp": 3,
    "mushroom": 43922
}


def decode_bytes(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)
    return df


def download_and_save(name, dataset_id):
    print(f"Downloading dataset '{name}' (id={dataset_id})...")
    dataset = openml.datasets.get_dataset(dataset_id, download_all_files=True)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe"
    )

    df = pd.concat([X, y.rename("target")], axis=1)
    df = decode_bytes(df)

    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {name}.csv  -> shape={df.shape}")


def main():
    for name, id_ in DATASETS.items():
        try:
            download_and_save(name, id_)
        except Exception as e:
            print(f"Error downloading {name}: {e}")


if __name__ == "__main__":
    main()
