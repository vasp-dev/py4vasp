#!/usr/bin/env python3
# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Assign Hugo weights to markdown files in the documentation build output.

This script walks through the Hugo markdown files and assigns weights based on:
- X (always 1): Top-level multiplier (10000)
- Y: Module index (100) - alphabetically sorted modules
- Z: File index (1) - alphabetically sorted files within each module

Formula: weight = 10000 * X + 100 * Y + Z

Examples:
- docs/_build/hugo/hugo/index.md -> weight = 10000 (special case)
- docs/_build/hugo/hugo/calculation.md -> weight = 10100
- docs/_build/hugo/hugo/calculation/band.md -> weight = 10101
- docs/_build/hugo/hugo/calculation/bandgap.md -> weight = 10102
"""

import re
from pathlib import Path
from typing import Dict, List


def get_module_and_file_mappings(hugo_dir: Path) -> Dict[str, int]:
    """
    Build a mapping of file paths to weights.
    
    Returns a dictionary where keys are relative paths and values are weights.
    """
    weights = {}
    
    # Special case: index.md at root gets weight 10000
    index_file = hugo_dir / "index.md"
    if index_file.exists():
        weights["index.md"] = 10000
    else:
        index_file = hugo_dir / "_index.md"
        if index_file.exists():
            weights["_index.md"] = 10000
    
    # Get all subdirectories (modules) and sort them alphabetically
    modules = sorted([d for d in hugo_dir.iterdir() if d.is_dir()])
    
    for module_idx, module_dir in enumerate(modules, start=1):
        module_name = module_dir.name
        
        # Module index file (e.g., calculation.md in hugo/calculation/)
        # This should actually be at hugo/calculation.md, not hugo/calculation/calculation.md
        module_index = hugo_dir / f"{module_name}.md"
        if module_index.exists():
            # Module gets weight: 10000 + 100 * module_idx
            weights[f"{module_name}.md"] = 10000 + 100 * module_idx
        
        # Get all .md files in the module directory and sort alphabetically
        module_files = sorted([f for f in module_dir.glob("*.md")])
        
        for file_idx, md_file in enumerate(module_files, start=1):
            relative_path = f"{module_name}/{md_file.name}"
            # File gets weight: 10000 + 100 * module_idx + file_idx
            weights[relative_path] = 10000 + 100 * module_idx + file_idx
    
    return weights


def assign_weights(hugo_dir: Path, dry_run: bool = False):
    """
    Assign weights to all markdown files in the Hugo output directory.
    
    Parameters
    ----------
    hugo_dir : Path
        Path to the hugo directory (e.g., docs/_build/hugo/hugo)
    dry_run : bool
        If True, print what would be done without modifying files
    """
    hugo_dir = Path(hugo_dir)
    
    if not hugo_dir.exists():
        raise ValueError(f"Hugo directory does not exist: {hugo_dir}")
    
    # Get weight mappings
    weights = get_module_and_file_mappings(hugo_dir)
    
    # Pattern to match the weight placeholder
    weight_pattern = re.compile(r'^weight = HUGO_WEIGHT_PLACEHOLDER$', re.MULTILINE)
    
    files_updated = 0
    
    for relative_path, weight in weights.items():
        md_file = hugo_dir / relative_path
        
        if not md_file.exists():
            continue
        
        # Read the file
        content = md_file.read_text(encoding='utf-8')
        
        # Check if placeholder exists
        if 'HUGO_WEIGHT_PLACEHOLDER' not in content:
            continue
        
        # Replace placeholder with actual weight
        new_content = weight_pattern.sub(f'weight = {weight}', content)
        
        if dry_run:
            print(f"Would update {relative_path}: weight = {weight}")
        else:
            md_file.write_text(new_content, encoding='utf-8')
            print(f"Updated {relative_path}: weight = {weight}")
        
        files_updated += 1
    
    print(f"\n{'Would update' if dry_run else 'Updated'} {files_updated} file(s)")


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Assign Hugo weights to markdown files"
    )
    parser.add_argument(
        "hugo_dir",
        type=Path,
        help="Path to the Hugo output directory (e.g., docs/_build/hugo/hugo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files",
    )
    
    args = parser.parse_args()
    
    assign_weights(args.hugo_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
