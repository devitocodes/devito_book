#!/usr/bin/env python3
"""
Resolve @@@CODE fromto: directives by finding actual line ranges in source files.

This script scans source code files for the patterns specified in fromto: directives
and generates a mapping of (file, pattern_start, pattern_end) -> (line_start, line_end).

Usage:
    python resolve_code_ranges.py --input doc/.src/chapters/ --code-base . --output code_ranges.json
"""

import argparse
import json
import os
import re
from pathlib import Path


def find_pattern_line(lines: list, pattern: str, start_line: int = 0) -> int | None:
    """Find the line number where a pattern first appears."""
    if not pattern or pattern == '_':
        return None

    # Handle special patterns
    if pattern.startswith('def ') or pattern.startswith('class '):
        # For function/class definitions, match the start
        for i, line in enumerate(lines[start_line:], start=start_line):
            if line.strip().startswith(pattern.strip()):
                return i
    else:
        # General pattern matching
        for i, line in enumerate(lines[start_line:], start=start_line):
            if pattern in line:
                return i

    return None


def extract_code_range(filepath: str, from_pattern: str, to_pattern: str, code_base: str = '.') -> dict:
    """Extract the line range for a fromto: directive."""
    full_path = os.path.join(code_base, filepath)

    # Handle src-vib -> src/vib conversion
    if not os.path.exists(full_path):
        alt_path = filepath.replace('src-', 'src/')
        full_path = os.path.join(code_base, alt_path)

    if not os.path.exists(full_path):
        return {
            'filepath': filepath,
            'error': f'File not found: {full_path}',
            'from_pattern': from_pattern,
            'to_pattern': to_pattern
        }

    try:
        with open(full_path, encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return {
            'filepath': filepath,
            'error': str(e),
            'from_pattern': from_pattern,
            'to_pattern': to_pattern
        }

    result = {
        'filepath': filepath,
        'resolved_path': full_path,
        'from_pattern': from_pattern,
        'to_pattern': to_pattern,
        'total_lines': len(lines)
    }

    # Find start line
    if from_pattern and from_pattern != '_':
        start_line = find_pattern_line(lines, from_pattern)
        if start_line is not None:
            result['start_line'] = start_line + 1  # 1-indexed
        else:
            result['start_line_error'] = f'Pattern not found: {from_pattern}'
            result['start_line'] = 1
    else:
        result['start_line'] = 1

    # Find end line
    if to_pattern and to_pattern != '_':
        search_start = result.get('start_line', 1) - 1
        end_line = find_pattern_line(lines, to_pattern, search_start)
        if end_line is not None:
            result['end_line'] = end_line + 1  # 1-indexed, inclusive
        else:
            result['end_line_error'] = f'Pattern not found: {to_pattern}'
            result['end_line'] = len(lines)
    else:
        result['end_line'] = len(lines)

    # Extract the code
    start = result['start_line'] - 1
    end = result['end_line']
    result['code'] = ''.join(lines[start:end])
    result['line_count'] = end - start

    return result


def scan_doconce_files(input_dir: str) -> list:
    """Scan DocOnce files for @@@CODE directives."""
    directives = []
    input_path = Path(input_dir)

    for do_file in input_path.rglob('*.do.txt'):
        with open(do_file, encoding='utf-8') as f:
            content = f.read()

        # Pattern: @@@CODE filepath fromto: pattern_start@pattern_end
        pattern = r'@@@CODE\s+(\S+)(?:\s+fromto:\s*([^@\n]+)@([^\n]+))?'

        for match in re.finditer(pattern, content):
            filepath = match.group(1).strip()
            from_pattern = match.group(2).strip() if match.group(2) else ''
            to_pattern = match.group(3).strip() if match.group(3) else ''

            directives.append({
                'source_file': str(do_file),
                'filepath': filepath,
                'from_pattern': from_pattern,
                'to_pattern': to_pattern
            })

    return directives


def main():
    parser = argparse.ArgumentParser(
        description='Resolve @@@CODE fromto: directives to line ranges'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Directory containing DocOnce files')
    parser.add_argument('--code-base', '-b', default='.',
                        help='Base directory for code file resolution')
    parser.add_argument('--output', '-o', required=True,
                        help='Output JSON file for code ranges')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose output')

    args = parser.parse_args()

    # Scan for directives
    print(f"Scanning {args.input} for @@@CODE directives...")
    directives = scan_doconce_files(args.input)
    print(f"Found {len(directives)} @@@CODE directives")

    # Resolve each directive
    results = []
    errors = []

    for directive in directives:
        if args.verbose:
            print(f"Resolving: {directive['filepath']}")

        result = extract_code_range(
            directive['filepath'],
            directive['from_pattern'],
            directive['to_pattern'],
            args.code_base
        )
        result['source_file'] = directive['source_file']

        if 'error' in result or 'start_line_error' in result or 'end_line_error' in result:
            errors.append(result)
        else:
            results.append(result)

    # Save results
    output = {
        'code_base': args.code_base,
        'total_directives': len(directives),
        'resolved': len(results),
        'errors': len(errors),
        'ranges': results,
        'error_details': errors
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"  Resolved: {len(results)}")
    print(f"  Errors: {len(errors)}")

    if errors and args.verbose:
        print("\nErrors:")
        for err in errors:
            print(f"  {err['filepath']}: {err.get('error', 'pattern not found')}")


if __name__ == '__main__':
    main()
