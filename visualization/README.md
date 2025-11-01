# Matrix Multiplication Visualizations

Interactive web-based demonstrations of matrix multiplication algorithms and their performance characteristics.

## Contents

- **index.html** - Interactive matrix multiplication demo (open in browser)
- **matmul_animations.py** - Python script to generate visualization media

## Generating Media Files

The `media/` directory contains generated SVG animations and videos. These files are excluded from version control (see `.gitignore`) but can be regenerated when needed.

To regenerate visualization media:

```bash
python3 visualization/matmul_animations.py
```

This will create the `media/` directory and populate it with:
- SVG animations showing matrix multiplication steps
- Video files demonstrating cache effects
- Performance comparison visualizations

## Requirements

For media generation:
```bash
pip3 install matplotlib numpy
```

The interactive HTML demo (`index.html`) has no dependencies—just open it in any modern web browser!
