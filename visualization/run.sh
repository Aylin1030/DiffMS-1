#!/bin/bash
# Script to generate images and launch the Streamlit app

echo "🔬 Molecule Visualization Tool"
echo "=============================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Check if images have been generated
if [ ! -d "visualization/molecule_images" ] || [ ! -f "visualization/molecule_images/metadata.csv" ]; then
    echo "📸 Generating molecule images..."
    echo ""
    uv run python visualization/generate_images.py
    echo ""
    echo "✅ Image generation complete!"
    echo ""
else
    echo "✅ Molecule images already exist"
    echo ""
fi

# Launch Streamlit app
echo "🚀 Launching Streamlit app..."
echo "   The app will open in your browser at http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

cd visualization
uv run streamlit run app.py

