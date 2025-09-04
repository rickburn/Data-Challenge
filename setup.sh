#!/bin/bash
# Setup script for Lending Club Data Challenge
# Run with: chmod +x setup.sh && ./setup.sh

set -e  # Exit on any error

echo "🚀 Setting up Lending Club Data Challenge environment..."

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📋 Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) ]]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔨 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing development dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Set up pre-commit hooks
echo "🎣 Setting up pre-commit hooks..."
pre-commit install

# Run initial tests to verify installation
echo "🧪 Running verification tests..."
pytest tests/unit/test_data_models.py -v

# Check code quality
echo "✨ Checking code quality..."
pre-commit run --all-files || echo "⚠️  Some pre-commit checks failed (this is normal for first run)"

# Test package import
echo "📋 Testing package import..."
python -c "from lending_club.models.data_models import LoanApplication; print('✅ Package imported successfully!')"

echo ""
echo "🎉 Setup complete! You can now:"
echo "   - Activate the environment: source venv/bin/activate"
echo "   - Run tests: pytest"
echo "   - Check code quality: pre-commit run --all-files"
echo "   - Start developing: see docs/GETTING_STARTED.md"
echo ""
echo "📚 Next steps:"
echo "   1. Review docs/REQUIREMENTS.md for project specifications"
echo "   2. Follow docs/GETTING_STARTED.md for development workflow"
echo "   3. Check data/ directory for available datasets"
