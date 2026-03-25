import sys
import pandas as pd
from typing import Optional
from rdkit import Chem


def canonicalize_smiles(s: object) -> Optional[str]:
    """Return RDKit canonical SMILES; None if parse fails."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    smi = str(s).strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def filter_valid_smiles(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Keep only valid SMILES"""
    def _is_valid(x):
        if not isinstance(x, str):
            return False
        x = x.strip()
        if not x:
            return False
        return Chem.MolFromSmiles(x) is not None

    mask = df[smiles_col].apply(_is_valid)
    return df[mask].copy()


def process_file(input_csv: str, smiles_col: str = "Smiles") -> pd.DataFrame:
    """Main processing pipeline"""
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")
    df = filter_valid_smiles(df, smiles_col)
    print(f"Valid SMILES: {len(df)}")
    df = df[df[smiles_col].str.count(r"\*") == 4]
    print(f"Polymer (*=4): {len(df)}")
    df["Smiles_canonical"] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["Smiles_canonical"])
    df = df.drop_duplicates(subset=["Smiles_canonical"]).reset_index(drop=True)
    print(f"Unique canonical SMILES: {len(df)}")

    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python xx.py input.csv [smiles_column]")
        sys.exit(1)

    input_csv = sys.argv[1]
    smiles_col = sys.argv[2] if len(sys.argv) > 2 else "Smiles"

    output_csv = input_csv.replace(".csv", "_valid_unique_polymer.csv")

    df_out = process_file(input_csv, smiles_col)

    df_out.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    main()
