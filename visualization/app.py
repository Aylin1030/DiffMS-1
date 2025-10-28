"""Streamlit app for molecule visualization and exploration."""
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import logging

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(metadata_path: Path) -> pd.DataFrame:
    """Load metadata from CSV file."""
    if not metadata_path.exists():
        st.error(f"Metadata file not found: {metadata_path}")
        st.info("Please run generate_images.py first to generate molecule images.")
        return pd.DataFrame()
    
    df = pd.read_csv(metadata_path)
    return df


def calculate_mol_properties(smiles: str) -> tuple[dict, str | None]:
    """Calculate molecular properties from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Tuple of (properties dict, error message or None)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}, "⚠️ Unable to parse SMILES - likely contains valence errors"
        
        return {
            'Molecular Weight': f"{Descriptors.MolWt(mol):.2f}",
            'LogP': f"{Descriptors.MolLogP(mol):.2f}",
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            'TPSA': f"{Descriptors.TPSA(mol):.2f}",
        }, None
    except Exception as e:
        logger.error(f"Error calculating properties for {smiles}: {e}")
        return {}, f"⚠️ Error: {str(e)}"


def generate_mol_image_on_fly(smiles: str, size: tuple[int, int] = (300, 300)) -> Image.Image | None:
    """Generate molecule image on the fly."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Molecule Prediction Viewer",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Molecule Prediction Viewer")
    st.markdown("---")
    
    # Load data
    metadata_path = Path(__file__).parent / 'molecule_images' / 'metadata.csv'
    df = load_data(metadata_path)
    
    if df.empty:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    spectrum_ids = sorted(df['spectrum_id'].unique())
    selected_spectrum = st.sidebar.selectbox(
        "Select Spectrum ID",
        options=['All'] + list(spectrum_ids),
        index=0
    )
    
    rank_filter = st.sidebar.slider(
        "Rank Range",
        min_value=int(df['rank'].min()),
        max_value=int(df['rank'].max()),
        value=(int(df['rank'].min()), int(df['rank'].max()))
    )
    
    valid_only = st.sidebar.checkbox("Show Valid Molecules Only", value=False)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_spectrum != 'All':
        filtered_df = filtered_df[filtered_df['spectrum_id'] == selected_spectrum]
    
    filtered_df = filtered_df[
        (filtered_df['rank'] >= rank_filter[0]) & 
        (filtered_df['rank'] <= rank_filter[1])
    ]
    
    if valid_only:
        filtered_df = filtered_df[filtered_df['valid'] == True]
    
    # Display statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Predictions", len(filtered_df))
    with col2:
        st.metric("Unique Spectra", filtered_df['spectrum_id'].nunique())
    with col3:
        valid_count = filtered_df['valid'].sum()
        st.metric("Valid Molecules", f"{valid_count} ({valid_count/len(filtered_df)*100:.1f}%)")
    with col4:
        parseable = filtered_df['image_path'].notna().sum()
        st.metric("Parseable SMILES", f"{parseable} ({parseable/len(filtered_df)*100:.1f}%)")
    with col5:
        avg_atoms = filtered_df['num_atoms'].mean()
        st.metric("Avg Atoms", f"{avg_atoms:.1f}")
    
    st.markdown("---")
    
    # Display mode selection
    display_mode = st.radio(
        "Display Mode",
        options=["Grid View", "Detailed View"],
        horizontal=True
    )
    
    if display_mode == "Grid View":
        # Grid view
        st.subheader("Molecule Grid")
        
        # Number of columns
        cols_per_row = st.slider("Columns per row", 2, 5, 3)
        
        # Create grid
        rows = []
        current_row = []
        
        for idx, row in filtered_df.iterrows():
            if len(current_row) == cols_per_row:
                rows.append(current_row)
                current_row = []
            current_row.append(row)
        
        if current_row:
            rows.append(current_row)
        
        # Display grid
        for row_data in rows:
            cols = st.columns(cols_per_row)
            for col, data in zip(cols, row_data):
                with col:
                    st.markdown(f"**Spectrum {data['spectrum_id']} - Rank {data['rank']}**")
                    
                    # Try to load saved image first, otherwise generate on the fly
                    if pd.notna(data['image_path']) and Path(data['image_path']).exists():
                        img = Image.open(data['image_path'])
                        st.image(img, use_container_width=True)
                    else:
                        # Try to generate on the fly
                        img = generate_mol_image_on_fly(data['smiles'], size=(300, 300))
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.error("❌ Cannot parse SMILES")
                            st.caption("Likely contains valence errors")
                    
                    st.caption(f"Atoms: {data['num_atoms']} | Valid: {'✅' if data['valid'] else '❌'}")
                    
                    with st.expander("SMILES"):
                        st.code(data['smiles'], language=None)
    
    else:
        # Detailed view
        st.subheader("Detailed Molecule View")
        
        # Select molecule
        options = [
            f"Spectrum {row['spectrum_id']} - Rank {row['rank']}" 
            for _, row in filtered_df.iterrows()
        ]
        
        if not options:
            st.warning("No molecules match the current filters.")
            st.stop()
        
        selected_idx = st.selectbox("Select a molecule", range(len(options)), format_func=lambda x: options[x])
        selected_row = filtered_df.iloc[selected_idx]
        
        # Display molecule
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Molecule Structure")
            
            # Try to load saved image first, otherwise generate on the fly
            if pd.notna(selected_row['image_path']) and Path(selected_row['image_path']).exists():
                img = Image.open(selected_row['image_path'])
                st.image(img, use_container_width=True)
            else:
                # Try to generate on the fly
                img = generate_mol_image_on_fly(selected_row['smiles'], size=(400, 400))
                if img:
                    st.image(img, use_container_width=True)
                    st.info("💡 Image generated on-the-fly")
                else:
                    st.error("❌ Cannot parse SMILES")
                    st.warning("""
                    This SMILES string contains chemical errors (e.g., valence violations).
                    The diffusion model generated an invalid molecule structure.
                    """)
            
            st.markdown("### SMILES String")
            st.code(selected_row['smiles'], language=None)
        
        with col2:
            st.markdown("### Prediction Details")
            details = {
                "Spectrum ID": selected_row['spectrum_id'],
                "Rank": selected_row['rank'],
                "Number of Atoms": selected_row['num_atoms'],
                "Valid": "✅ Yes" if selected_row['valid'] else "❌ No",
                "Total Candidates": selected_row['total_candidates']
            }
            
            for key, value in details.items():
                st.markdown(f"**{key}:** {value}")
            
            st.markdown("### Molecular Properties")
            props, error = calculate_mol_properties(selected_row['smiles'])
            
            if error:
                st.warning(error)
            elif props:
                for key, value in props.items():
                    st.markdown(f"**{key}:** {value}")
            else:
                st.warning("Could not calculate molecular properties")
    
    # Data table at bottom
    st.markdown("---")
    st.subheader("📊 Data Table")
    
    display_df = filtered_df[['spectrum_id', 'rank', 'smiles', 'num_atoms', 'valid', 'total_candidates']].copy()
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download option
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=display_df.to_csv(index=False).encode('utf-8'),
        file_name=f"filtered_predictions_{selected_spectrum}.csv",
        mime="text/csv"
    )


if __name__ == '__main__':
    main()

