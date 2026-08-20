"""
Utility functions for drug data processing.

Author: Research Team
Date: 2026
"""

from typing import Dict, Optional
import requests
from pathlib import Path


def fetch_smiles_from_drugbank(drugbank_id: str) -> Optional[str]:
    """
    Fetch SMILES string from DrugBank (requires API key).
    
    Note: This is a placeholder. In practice, you may need:
    1. DrugBank API credentials
    2. Or use ChEMBL/PubChem instead
    
    Args:
        drugbank_id: DrugBank ID (e.g., 'DB00843')
    
    Returns:
        SMILES string or None if not found
    """
    # Placeholder - actual implementation would require DrugBank API
    # For now, we'll use hardcoded SMILES or ChEMBL
    return None


def fetch_smiles_from_chembl(chembl_id: str) -> Optional[str]:
    """
    Fetch SMILES string from ChEMBL database.
    
    Args:
        chembl_id: ChEMBL ID (e.g., 'CHEMBL90')
    
    Returns:
        SMILES string or None if not found
    """
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            smiles = data.get('molecule_structures', {}).get('canonical_smiles')
            return smiles
        else:
            print(f"ChEMBL API returned status {response.status_code} for {chembl_id}")
            return None
    
    except Exception as e:
        print(f"Error fetching SMILES from ChEMBL for {chembl_id}: {str(e)}")
        return None


def fetch_smiles_from_pubchem(compound_name: str) -> Optional[str]:
    """
    Fetch SMILES string from PubChem database.
    
    Args:
        compound_name: Compound name (e.g., 'Donepezil')
    
    Returns:
        SMILES string or None if not found
    """
    try:
        # First, search for CID
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            smiles_list = data.get('PropertyTable', {}).get('Properties', [])
            if smiles_list:
                return smiles_list[0].get('CanonicalSMILES')
        
        return None
    
    except Exception as e:
        print(f"Error fetching SMILES from PubChem for {compound_name}: {str(e)}")
        return None


def validate_smiles(smiles: str) -> bool:
    """
    Basic validation of SMILES string.
    
    Args:
        smiles: SMILES string
    
    Returns:
        True if SMILES appears valid
    """
    if not smiles or len(smiles) == 0:
        return False
    
    # Basic checks
    # SMILES should contain alphanumeric characters and some special characters
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]=+-#@$%&*./\\')
    if not all(c in valid_chars or c.isspace() for c in smiles):
        return False
    
    # Should have at least one letter
    if not any(c.isalpha() for c in smiles):
        return False
    
    return True



