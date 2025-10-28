"""Generate molecule images from SMILES strings."""
import csv
from pathlib import Path
from typing import List, Optional
import logging

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import pandas as pd

from models import MoleculePrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions(tsv_file: Path) -> List[MoleculePrediction]:
    """Load predictions from TSV file.
    
    Args:
        tsv_file: Path to the TSV file
        
    Returns:
        List of MoleculePrediction objects
    """
    predictions = []
    
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                pred = MoleculePrediction(
                    spectrum_id=int(row['spectrum_id']),
                    rank=int(row['rank']),
                    smiles=row['smiles'],
                    num_atoms=int(row['num_atoms']),
                    valid=row['valid'] == 'True',
                    total_candidates=int(row['total_candidates'])
                )
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error parsing row: {row}, error: {e}")
                
    return predictions


def generate_molecule_image(smiles: str, img_size: tuple[int, int] = (400, 400)) -> Optional[Image.Image]:
    """Generate molecule image from SMILES string.
    
    Args:
        smiles: SMILES string
        img_size: Image size as (width, height)
        
    Returns:
        PIL Image object or None if failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
            
        img = Draw.MolToImage(mol, size=img_size)
        return img
    except Exception as e:
        logger.error(f"Error generating image for SMILES {smiles}: {e}")
        return None


def generate_all_images(
    tsv_file: Path,
    output_dir: Path,
    img_size: tuple[int, int] = (400, 400)
) -> pd.DataFrame:
    """Generate images for all molecules in the TSV file.
    
    Args:
        tsv_file: Path to input TSV file
        output_dir: Directory to save images
        img_size: Image size as (width, height)
        
    Returns:
        DataFrame with predictions and image paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = load_predictions(tsv_file)
    logger.info(f"Loaded {len(predictions)} predictions")
    
    results = []
    
    for pred in predictions:
        img = generate_molecule_image(pred.smiles, img_size)
        
        # Save image
        img_filename = f"spectrum_{pred.spectrum_id}_rank_{pred.rank}.png"
        img_path = output_dir / img_filename
        
        if img:
            img.save(img_path)
            logger.info(f"Saved image: {img_path}")
        else:
            img_path = None
            logger.warning(f"Failed to generate image for spectrum {pred.spectrum_id}, rank {pred.rank}")
        
        results.append({
            'spectrum_id': pred.spectrum_id,
            'rank': pred.rank,
            'smiles': pred.smiles,
            'num_atoms': pred.num_atoms,
            'valid': pred.valid,
            'total_candidates': pred.total_candidates,
            'image_path': str(img_path) if img_path else None
        })
    
    df = pd.DataFrame(results)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.csv'
    df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")
    
    return df


if __name__ == '__main__':
    # Default paths
    tsv_file = Path(__file__).parent.parent / 'results' / 'predictions_all_candidates.tsv'
    output_dir = Path(__file__).parent / 'molecule_images'
    
    logger.info(f"Processing {tsv_file}")
    logger.info(f"Output directory: {output_dir}")
    
    df = generate_all_images(tsv_file, output_dir)
    
    logger.info(f"Generated {len(df)} molecule images")
    logger.info(f"Success rate: {df['image_path'].notna().sum() / len(df) * 100:.1f}%")

