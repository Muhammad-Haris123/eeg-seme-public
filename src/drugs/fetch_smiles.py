"""
Drug SMILES Fetching Module.

This module fetches SMILES strings for target drugs from public databases:
- ChEMBL
- PubChem
- DrugBank (placeholder)

Drugs of interest:
- Donepezil
- Memantine

Author: Research Team
Date: 2026
"""

import json
import csv
from pathlib import Path
from typing import Dict, Optional

from ..utils.config import TARGET_DRUGS, DRUG_EMBEDDINGS_DIR
from .utils import (
    fetch_smiles_from_chembl,
    fetch_smiles_from_pubchem,
    validate_smiles
)


class SMILESFetcher:
    """
    Fetcher for drug SMILES strings from multiple sources.
    """
    
    def __init__(self):
        """Initialize SMILES fetcher."""
        self.target_drugs = TARGET_DRUGS.copy()
    
    def fetch_drug_smiles(
        self,
        drug_key: str,
        prefer_source: str = 'chembl'
    ) -> Optional[str]:
        """
        Fetch SMILES string for a drug.
        
        Args:
            drug_key: Drug key in TARGET_DRUGS ('donepezil' or 'memantine')
            prefer_source: Preferred source ('chembl', 'pubchem', 'drugbank')
        
        Returns:
            SMILES string or None if not found
        """
        if drug_key not in self.target_drugs:
            raise ValueError(f"Unknown drug: {drug_key}")
        
        drug_info = self.target_drugs[drug_key]
        smiles = None
        
        # Try ChEMBL first if preferred
        if prefer_source == 'chembl' and drug_info.get('chembl_id'):
            smiles = fetch_smiles_from_chembl(drug_info['chembl_id'])
        
        # Try PubChem
        if not smiles:
            smiles = fetch_smiles_from_pubchem(drug_info['name'])
        
        # Try ChEMBL as fallback
        if not smiles and prefer_source != 'chembl' and drug_info.get('chembl_id'):
            smiles = fetch_smiles_from_chembl(drug_info['chembl_id'])
        
        # Validate SMILES
        if smiles and validate_smiles(smiles):
            return smiles
        else:
            print(f"Warning: Invalid or missing SMILES for {drug_key}")
            return None
    
    def fetch_all_smiles(self) -> Dict[str, str]:
        """
        Fetch SMILES strings for all target drugs.
        
        Returns:
            Dictionary mapping drug keys to SMILES strings
        """
        all_smiles = {}
        
        for drug_key in self.target_drugs.keys():
            print(f"Fetching SMILES for {self.target_drugs[drug_key]['name']}...")
            smiles = self.fetch_drug_smiles(drug_key)
            
            if smiles:
                all_smiles[drug_key] = smiles
                print(f"  Found: {smiles[:50]}...")
            else:
                print(f"  Failed to fetch SMILES")
        
        return all_smiles
    
    def save_smiles(
        self,
        smiles_dict: Dict[str, str],
        output_dir: Optional[Path] = None
    ):
        """
        Save SMILES strings to file.
        
        Args:
            smiles_dict: Dictionary mapping drug keys to SMILES
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = DRUG_EMBEDDINGS_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        output_path = output_dir / "drug_smiles.json"
        with open(output_path, 'w') as f:
            json.dump(smiles_dict, f, indent=2)
        
        print(f"Saved SMILES to: {output_path}")
        
        # Also save individual files
        for drug_key, smiles in smiles_dict.items():
            drug_path = output_dir / f"{drug_key}_smiles.txt"
            with open(drug_path, 'w') as f:
                f.write(smiles)
        
        return output_path


def load_smiles_from_csv(csv_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load SMILES strings from CSV file.
    
    Args:
        csv_path: Path to CSV file (defaults to DRUG_EMBEDDINGS_DIR / "drugs.csv")
    
    Returns:
        Dictionary mapping drug names (lowercase) to SMILES strings
    """
    if csv_path is None:
        csv_path = DRUG_EMBEDDINGS_DIR / "drugs.csv"
    
    smiles_dict = {}
    
    if csv_path.exists():
        print(f"Loading SMILES from CSV: {csv_path}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_name = row.get('drug_name', '').strip()
                smiles = row.get('smiles', '').strip()
                
                if drug_name and smiles:
                    # Convert to lowercase key for consistency
                    drug_key = drug_name.lower()
                    smiles_dict[drug_key] = smiles
                    print(f"  Loaded {drug_name}: {smiles[:50]}...")
    else:
        print(f"CSV file not found: {csv_path}")
    
    return smiles_dict


def fetch_and_save_smiles(csv_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load SMILES from CSV file or fetch from APIs if CSV not available.
    
    Args:
        csv_path: Path to CSV file (optional)
    
    Returns:
        Dictionary of drug SMILES
    """
    # Try loading from CSV first
    smiles_dict = load_smiles_from_csv(csv_path)
    
    # If CSV loading failed or empty, try fetching from APIs
    if not smiles_dict:
        print("CSV not available, fetching from APIs...")
        fetcher = SMILESFetcher()
        smiles_dict = fetcher.fetch_all_smiles()
    
    if smiles_dict:
        fetcher = SMILESFetcher()
        fetcher.save_smiles(smiles_dict)
    
    return smiles_dict


if __name__ == "__main__":
    # Example usage
    smiles = fetch_and_save_smiles()
    print(f"\nFetched SMILES for {len(smiles)} drugs")

