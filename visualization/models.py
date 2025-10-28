"""Data models for molecule visualization."""
from typing import Optional
from pydantic import BaseModel, Field


class MoleculePrediction(BaseModel):
    """Model for a single molecule prediction."""
    spectrum_id: int = Field(description="Spectrum ID")
    rank: int = Field(description="Rank of the prediction")
    smiles: str = Field(description="SMILES string")
    num_atoms: int = Field(description="Number of atoms")
    valid: bool = Field(description="Whether the molecule is valid")
    total_candidates: int = Field(description="Total number of candidates")
    
    class Config:
        frozen = False

