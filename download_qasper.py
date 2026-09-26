from pathlib import Path

from datasets import load_dataset

PARQUET_URL = (
    "https://huggingface.co/datasets/allenai/qasper/resolve/"
    "refs%2Fconvert%2Fparquet/qasper/validation/0000.parquet"
)


def load_qasper_validation():
    try:
        # Auto-converted Parquet branch (the main branch uses an unsupported script)
        return load_dataset(
            "allenai/qasper",
            "qasper",
            split="validation",
            revision="refs/convert/parquet",
        )
    except Exception as e:
        print(f"Loading from the Parquet branch failed ({e}), trying the direct file URL...")
        return load_dataset("parquet", data_files=PARQUET_URL, split="train")


def main():
    dataset = load_qasper_validation()
    print(f"Loaded {len(dataset)} papers\n")

    print(dataset[0])

    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "qasper_validation.parquet"
    dataset.to_parquet(str(output_path))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()