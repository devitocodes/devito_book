#!/usr/bin/env python3
"""
DocOnce to Quarto Converter

Converts DocOnce (.do.txt) files to Quarto (.qmd) format.

Usage:
    python doconce_to_quarto.py --input file.do.txt --output file.qmd [options]

Options:
    --input         Input DocOnce file
    --output        Output Quarto file
    --code-base     Base directory for @@@CODE file resolution
    --expand-mako   Expand Mako variables to static values (default: True)
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversionStats:
    """Track conversion statistics"""
    math_blocks: int = 0
    code_blocks: int = 0
    code_includes: int = 0
    figures: int = 0
    admonitions: int = 0
    citations: int = 0
    labels_converted: int = 0
    refs_converted: int = 0
    idx_removed: int = 0
    headers_converted: int = 0


class DocOnceToQuartoConverter:
    """Main converter class with all transformation methods"""

    def __init__(self, code_base: str | None = None, expand_mako: bool = True):
        self.code_base = code_base or "."
        self.expand_mako = expand_mako
        self.stats = ConversionStats()

        # Mako variable expansions (from mako_code.txt)
        self.mako_vars = {
            'src': 'https://github.com/hplgit/fdm-book/tree/master/src',
            'src_vib': 'https://github.com/hplgit/fdm-book/tree/master/src/vib',
            'src_wave': 'https://github.com/hplgit/fdm-book/tree/master/src/wave',
            'src_diffu': 'https://github.com/hplgit/fdm-book/tree/master/src/diffu',
            'src_trunc': 'https://github.com/hplgit/fdm-book/tree/master/src/trunc',
            'src_nonlin': 'https://github.com/hplgit/fdm-book/tree/master/src/nonlin',
            'src_advec': 'https://github.com/hplgit/fdm-book/tree/master/src/advec',
            'src_formulas': 'https://github.com/hplgit/fdm-book/tree/master/src/formulas',
            'src_softeng2': 'https://github.com/hplgit/fdm-book/tree/master/src/softeng2',
            'doc': 'http://hplgit.github.io/fdm-book/doc/pub',
            'doc_notes': 'http://hplgit.github.io/fdm-book/doc/pub',
        }

        # DocOnce code language mapping to Quarto
        self.lang_map = {
            'pycod': 'python',
            'pypro': 'python',
            'py': 'python',
            'python': 'python',
            'ipy': 'python',
            'cppcod': 'cpp',
            'cpppro': 'cpp',
            'cpp': 'cpp',
            'ccod': 'c',
            'cpro': 'c',
            'c': 'c',
            'fcod': 'fortran',
            'fpro': 'fortran',
            'fortran': 'fortran',
            'f': 'fortran',
            'shcod': 'bash',
            'shpro': 'bash',
            'sh': 'bash',
            'bash': 'bash',
            'sys': 'bash',
            'text': 'text',
            'dat': 'text',
            'txt': 'text',
            'latexcod': 'latex',
            'latex': 'latex',
            'htmlcod': 'html',
            'html': 'html',
            'do': 'text',
            'cod': 'python',  # generic code - default to python for this book
            'pro': 'python',  # generic program - default to python for this book
            '': 'python',     # no language specified - default to python
        }

        # Admonition type mapping
        self.admon_map = {
            'warning': 'warning',
            'notice': 'note',
            'question': 'tip',
            'summary': 'important',
            'block': 'note',
            'hint': 'tip',
            'quote': 'note',
        }

    def convert(self, content: str) -> str:
        """Run all conversion steps in order"""
        # Order matters - some conversions depend on others

        # 1. Handle includes first (flatten structure)
        content = self.resolve_includes(content)

        # 2. Expand Mako variables
        if self.expand_mako:
            content = self.convert_mako_variables(content)

        # 3. Remove TOC and split directives
        content = self.remove_toc_and_split(content)

        # 4. Convert headers (must come before label handling)
        content = self.convert_headers(content)

        # 5. Convert math blocks (must handle labels inside)
        content = self.convert_math_blocks(content)

        # 5b. Convert standalone \[...\] display math
        content = self.convert_bracket_math(content)

        # 6. Convert code blocks
        content = self.convert_code_blocks(content)

        # 7. Convert @@@CODE directives
        content = self.convert_code_includes(content)

        # 8. Convert figures
        content = self.convert_figures(content)

        # 9. Convert admonitions
        content = self.convert_admonitions(content)

        # 10. Convert citations
        content = self.convert_citations(content)

        # 11. Convert inline URLs
        content = self.convert_urls(content)

        # 12. Convert cross-references (labels and refs)
        content = self.convert_labels_and_refs(content)

        # 13. Remove index entries
        content = self.remove_index_entries(content)

        # 14. Convert lists
        content = self.convert_lists(content)

        # 15. Convert inline formatting
        content = self.convert_inline_formatting(content)

        # 16. Remove DocOnce comments
        content = self.remove_doconce_comments(content)

        # 17. Final cleanup
        content = self.cleanup(content)

        return content

    def resolve_includes(self, content: str) -> str:
        """Resolve # #include directives (flatten for now)"""
        # Pattern: # #include "filename"
        pattern = r'^#\s*#include\s+"([^"]+)"'

        def replace_include(match):
            filename = match.group(1)
            # For now, just leave a comment noting the include
            return f'<!-- Include: {filename} -->'

        return re.sub(pattern, replace_include, content, flags=re.MULTILINE)

    def convert_mako_variables(self, content: str) -> str:
        """Expand ${var} Mako variables to their values"""
        # Pattern: ${var_name}
        pattern = r'\$\{(\w+)\}'

        def replace_var(match):
            var_name = match.group(1)
            if var_name in self.mako_vars:
                return self.mako_vars[var_name]
            # Return as Quarto variable reference
            return f'{{{{< var {var_name} >}}}}'

        return re.sub(pattern, replace_var, content)

    def remove_toc_and_split(self, content: str) -> str:
        """Remove TOC: and !split directives"""
        content = re.sub(r'^TOC:.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^!split\s*$', '', content, flags=re.MULTILINE)
        return content

    def convert_headers(self, content: str) -> str:
        """Convert DocOnce header syntax to Markdown"""
        # DocOnce uses = signs for headers:
        # ======= Title ======= (chapter/h1)
        # ===== Title ===== (section/h2)
        # === Title === (subsection/h3)

        def replace_header(match):
            equals = match.group(1)
            title = match.group(2).strip()
            self.stats.headers_converted += 1

            # Count = signs to determine level
            eq_count = len(equals)
            if eq_count >= 7:
                level = 1  # Chapter
            elif eq_count >= 5:
                level = 2  # Section
            else:
                level = 3  # Subsection

            return '#' * level + ' ' + title

        # Match headers with = signs
        pattern = r'^(={3,})\s+(.+?)\s+=+\s*$'
        return re.sub(pattern, replace_header, content, flags=re.MULTILINE)

    def convert_math_blocks(self, content: str) -> str:
        """Convert !bt/!et math blocks to $$ $$ blocks"""
        # Pattern: !bt ... !et with potential labels inside

        def replace_math_block(match):
            self.stats.math_blocks += 1
            math_content = match.group(1)

            # Extract label if present (could be \label or label{} in DocOnce)
            label_match = re.search(r'\\?label\{([^}]+)\}', math_content)
            label_id = None
            if label_match:
                label_id = label_match.group(1)
                self.stats.labels_converted += 1
                # Remove the label from inside the math for now
                math_content = re.sub(r'\\?label\{[^}]+\}', '', math_content)

            # Clean up the math content
            math_content = math_content.strip()

            # Check if content has align environment - if so, DON'T wrap in $$
            # because align is already a display math environment
            has_align = r'\begin{align' in math_content

            # Remove equation wrappers (but keep align for multi-line equations)
            math_content = re.sub(r'\\begin\{equation\*?\}', '', math_content)
            math_content = re.sub(r'\\end\{equation\*?\}', '', math_content)
            # Also remove \[ and \] display math delimiters (we use $$ instead)
            math_content = re.sub(r'^\s*\\\[\s*$', '', math_content, flags=re.MULTILINE)
            math_content = re.sub(r'^\s*\\\]\s*$', '', math_content, flags=re.MULTILINE)
            # Handle inline \[ and \] at start/end of content
            math_content = re.sub(r'^\s*\\\[', '', math_content)
            math_content = re.sub(r'\\\]\s*$', '', math_content)

            # Handle \tp (thinspace period) - move it to end of last content line
            # and remove any standalone \tp lines
            lines = math_content.strip().split('\n')
            has_tp = False
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped == '\\tp':
                    has_tp = True
                elif stripped:  # non-empty line
                    cleaned_lines.append(line)

            # Reassemble, adding \tp to the last line if needed
            if cleaned_lines:
                math_content = '\n'.join(cleaned_lines)
                if has_tp:
                    math_content = math_content.rstrip() + ' \\tp'
            else:
                math_content = ''

            math_content = math_content.strip()

            # Remove multiple blank lines
            math_content = re.sub(r'\n\s*\n', '\n', math_content)

            # Build the result - don't wrap align in $$ since it's already display math
            if has_align:
                result = math_content
                # For align environments, add LaTeX \label inside the environment
                if label_id:
                    # Add \label just before \end{align}
                    result = re.sub(
                        r'(\\end\{align\*?\})',
                        r'\\label{' + label_id + r'}\n\1',
                        result
                    )
            else:
                result = f'$$\n{math_content}\n$$'
                # Add Quarto label after the closing $$ on same line
                if label_id:
                    # Convert label format: vib:ode1 -> #eq-vib-ode1
                    clean_id = label_id.replace(':', '-').replace('_', '-')
                    result += f' {{#eq-{clean_id}}}'

            return result

        # Match !bt ... !et blocks
        pattern = r'!bt\s*\n(.*?)\n!et'
        return re.sub(pattern, replace_math_block, content, flags=re.DOTALL)

    def convert_bracket_math(self, content: str) -> str:
        r"""Convert standalone \[...\] display math to $$...$$ blocks"""
        # Pattern: \[ ... \] on single line or multi-line
        # Handle single-line first (most common in this codebase)
        def replace_bracket_math(match):
            math_content = match.group(1).strip()
            return f'$$\n{math_content}\n$$'

        # Single-line: \[...\]
        content = re.sub(r'\\\[([^\]]+)\\\]', replace_bracket_math, content)

        # Multi-line: \[ on own line, then content, then \] on own line
        content = re.sub(
            r'^\s*\\\[\s*\n(.*?)\n\s*\\\]\s*$',
            lambda m: f'$$\n{m.group(1).strip()}\n$$',
            content,
            flags=re.DOTALL | re.MULTILINE
        )

        return content

    def convert_code_blocks(self, content: str) -> str:
        """Convert !bc/!ec code blocks to ``` ``` blocks"""

        def replace_code_block(match):
            self.stats.code_blocks += 1
            lang_code = match.group(1) or ''
            code_content = match.group(2)

            # Map the language
            lang = self.lang_map.get(lang_code.strip(), 'python')

            # Use static code blocks (not executable) - no curly braces
            return f'```{lang}\n{code_content}\n```'

        # Match !bc [lang] ... !ec blocks
        pattern = r'!bc\s*(\w*)\s*\n(.*?)\n!ec'
        return re.sub(pattern, replace_code_block, content, flags=re.DOTALL)

    def convert_code_includes(self, content: str) -> str:
        """Convert @@@CODE directives to Quarto includes"""

        def replace_code_include(match):
            self.stats.code_includes += 1
            filepath = match.group(1).strip()
            fromto = match.group(2) if match.lastindex >= 2 else None

            # Resolve relative path
            original_path = filepath
            if filepath.startswith('src-'):
                # Convert src-vib/file.py to src/vib/file.py
                filepath = filepath.replace('src-', 'src/')

            # Try to read the actual file
            full_path = Path(self.code_base) / filepath
            if not full_path.exists():
                # Try alternative path
                alt_path = Path(self.code_base) / original_path
                if alt_path.exists():
                    full_path = alt_path

            code_content = ""
            if full_path.exists():
                try:
                    with open(full_path, encoding='utf-8') as f:
                        lines = f.readlines()

                    # Parse fromto patterns if provided
                    if fromto:
                        parts = fromto.strip().split('@')
                        from_pattern = parts[0].strip() if len(parts) > 0 else ''
                        to_pattern = parts[1].strip() if len(parts) > 1 else ''

                        start_line = 0
                        end_line = len(lines)

                        # Find start pattern
                        if from_pattern and from_pattern != '_':
                            for i, line in enumerate(lines):
                                if from_pattern in line:
                                    start_line = i
                                    break

                        # Find end pattern
                        if to_pattern and to_pattern != '_':
                            for i, line in enumerate(lines[start_line:], start=start_line):
                                if to_pattern in line:
                                    end_line = i
                                    break

                        code_content = ''.join(lines[start_line:end_line])
                    else:
                        code_content = ''.join(lines)

                    # Remove trailing whitespace from each line
                    code_content = '\n'.join(line.rstrip() for line in code_content.split('\n'))
                    code_content = code_content.strip()

                except Exception as e:
                    code_content = f"# Error reading file: {e}"
            else:
                code_content = f"# File not found: {filepath}"

            # Determine language from file extension
            lang = 'python'
            if filepath.endswith('.cpp') or filepath.endswith('.cc'):
                lang = 'cpp'
            elif filepath.endswith('.c'):
                lang = 'c'
            elif filepath.endswith('.f') or filepath.endswith('.f90'):
                lang = 'fortran'
            elif filepath.endswith('.sh'):
                lang = 'bash'

            # Use static code blocks (not executable) - no curly braces
            result = f'```{lang}\n{code_content}\n```'
            return result

        # Match @@@CODE file fromto: pattern@pattern
        pattern = r'@@@CODE\s+(\S+)(?:\s+fromto:\s*(.+?))?(?=\n|$)'
        return re.sub(pattern, replace_code_include, content)

    def convert_figures(self, content: str) -> str:
        """Convert FIGURE: directives to Quarto figure syntax"""

        def replace_figure(match):
            self.stats.figures += 1
            path = match.group(1).strip()
            options = match.group(2) or ''
            caption = match.group(3).strip() if match.lastindex >= 3 else ''

            # Extract label from caption if present
            label_match = re.search(r'label\{([^}]+)\}', caption)
            label_id = None
            if label_match:
                label_id = label_match.group(1)
                self.stats.labels_converted += 1
                caption = re.sub(r'\s*label\{[^}]+\}', '', caption)

            # Clean up path - convert fig-vib to fig
            path = re.sub(r'fig-\w+/', 'fig/', path)

            # Build Quarto figure
            if caption:
                result = f'![{caption}]({path})'
            else:
                result = f'![]({path})'

            # Add attributes
            attrs = []
            if label_id:
                quarto_label = self._convert_label(label_id, 'fig')
                attrs.append(quarto_label)

            # Extract width if specified
            width_match = re.search(r'width=(\d+)', options)
            if width_match:
                attrs.append(f'width="{width_match.group(1)}px"')

            if attrs:
                result += '{' + ' '.join(attrs) + '}'

            return result

        # Match FIGURE: [path, options] caption label{...}
        pattern = r'FIGURE:\s*\[([^\],]+)(?:,\s*([^\]]*))?\]\s*(.+?)(?=\n\n|\n[A-Z]|\Z)'
        return re.sub(pattern, replace_figure, content, flags=re.DOTALL)

    def convert_admonitions(self, content: str) -> str:
        """Convert !bwarning/!ewarning etc to Quarto callouts"""

        def replace_admonition(match):
            self.stats.admonitions += 1
            admon_type = match.group(1)
            title = match.group(2).strip() if match.group(2) else ''
            body = match.group(3).strip()

            # Map to Quarto callout type
            quarto_type = self.admon_map.get(admon_type, 'note')

            result = f':::{{.callout-{quarto_type}'
            if title:
                result += f' title="{title}"'
            result += '}\n'
            result += body + '\n'
            result += ':::'

            return result

        # Match !btype Title ... !etype
        # Types: warning, notice, question, summary, block, hint, quote
        pattern = r'!b(warning|notice|question|summary|block|hint|quote)\s*(.*?)\n(.*?)\n!e\1'
        return re.sub(pattern, replace_admonition, content, flags=re.DOTALL)

    def convert_citations(self, content: str) -> str:
        """Convert cite{key} to [@key] format"""

        def replace_citation(match):
            self.stats.citations += 1
            keys = match.group(1)
            # Handle multiple keys: cite{key1,key2}
            key_list = [k.strip() for k in keys.split(',')]
            return '[' + '; '.join(f'@{k}' for k in key_list) + ']'

        # Match cite{key} or cite{key1,key2}
        pattern = r'cite\{([^}]+)\}'
        return re.sub(pattern, replace_citation, content)

    def convert_urls(self, content: str) -> str:
        """Convert "text": "url" to [text](url) format"""
        # DocOnce inline URL: "Link text": "http://example.com"
        pattern = r'"([^"]+)":\s*"(https?://[^"]+)"'

        def replace_url(match):
            text = match.group(1)
            url = match.group(2)
            return f'[{text}]({url})'

        return re.sub(pattern, replace_url, content)

    def convert_labels_and_refs(self, content: str) -> str:
        """Convert remaining label{} and ref{} to Quarto format"""

        # Convert standalone labels (not in math - those are handled already)
        def replace_label(match):
            label_id = match.group(1)
            # Determine if this is a section label based on context
            # Section labels typically have patterns like vib:model1, sec:xxx
            self.stats.labels_converted += 1
            clean_id = label_id.replace(':', '-').replace('_', '-')
            return f'{{#sec-{clean_id}}}'

        # Convert ref{} to @ref-
        def replace_ref(match):
            label_id = match.group(1)
            self.stats.refs_converted += 1
            quarto_ref = self._convert_ref(label_id)
            return quarto_ref

        # Replace labels first (often appear after headers)
        # Only match standalone label{} not \label{} (already processed in math)
        content = re.sub(r'(?<!\\)\blabel\{([^}]+)\}', replace_label, content)

        # Replace refs
        content = re.sub(r'\bref\{([^}]+)\}', replace_ref, content)

        return content

    def _convert_label(self, label_id: str, prefix: str) -> str:
        """Convert DocOnce label to Quarto format"""
        # vib:ode1 -> #eq-vib-ode1
        clean_id = label_id.replace(':', '-').replace('_', '-')
        return f'#{prefix}-{clean_id}'

    def _convert_ref(self, label_id: str) -> str:
        """Convert DocOnce ref to Quarto format"""
        # Determine prefix based on label pattern
        clean_id = label_id.replace(':', '-').replace('_', '-')
        label_lower = label_id.lower()

        # Try to guess the type from the label naming conventions
        if any(x in label_lower for x in ['fig:', 'fig-', 'figure']):
            return f'@fig-{clean_id}'
        elif any(x in label_lower for x in ['sec:', 'chap:', 'app:', 'model', 'impl', 'verify']):
            return f'@sec-{clean_id}'
        elif any(x in label_lower for x in ['tab:', 'table']):
            return f'@tbl-{clean_id}'
        elif any(x in label_lower for x in ['exer:', 'problem']):
            return f'@sec-{clean_id}'
        else:
            # Default to equation reference for vib:ode1 style labels
            return f'@eq-{clean_id}'

    def remove_index_entries(self, content: str) -> str:
        """Remove idx{} index entries"""

        def count_and_remove(match):
            self.stats.idx_removed += 1
            return ''

        # Match idx{term} or idx{term1} idx{term2} on same line
        pattern = r'\s*idx\{[^}]+\}'
        return re.sub(pattern, count_and_remove, content)

    def convert_lists(self, content: str) -> str:
        """Convert DocOnce list syntax to Markdown"""
        # DocOnce uses:
        # o item (numbered)
        # * item (bullet)
        # - item (also bullet)

        # Convert numbered lists: lines starting with 'o '
        content = re.sub(r'^(\s*)o\s+', r'\g<1>1. ', content, flags=re.MULTILINE)

        # Bullet lists with * are already Markdown compatible
        # Convert - at start of line to * for consistency
        # (but be careful not to match horizontal rules or other uses)
        # content = re.sub(r'^(\s*)-\s+', r'\g<1>* ', content, flags=re.MULTILINE)

        return content

    def convert_inline_formatting(self, content: str) -> str:
        """Convert DocOnce inline formatting to Markdown"""
        # *emphasis* -> *emphasis* (same)
        # _bold_ -> **bold** (different!)
        # `code` -> `code` (same)

        # Convert _bold_ to **bold** (but not file_name_with_underscores)
        # This is tricky - DocOnce _text_ is bold, but _ is common in code
        # For safety, only convert when there's clearly text between
        content = re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)', r'**\1**', content)

        return content

    def remove_doconce_comments(self, content: str) -> str:
        """Remove DocOnce comment lines (starting with # but not ##)"""
        # DocOnce comments are lines starting with # followed by content
        # Need to preserve: ## headers, # #include directives
        # Remove: #comment, #+ continuation comments, etc.
        lines = content.split('\n')
        result = []
        for line in lines:
            stripped = line.lstrip()
            # Skip pure DocOnce comment lines (but keep ## headers)
            if stripped.startswith('#') and not stripped.startswith('##'):
                # Check if it's a special directive we want to keep
                if stripped.startswith('# #include'):
                    result.append(line)
                # Otherwise it's a comment - skip it
                continue
            result.append(line)
        return '\n'.join(result)

    def cleanup(self, content: str) -> str:
        """Final cleanup of converted content"""

        # Remove [hpl: ...] author notes
        content = re.sub(r'\[hpl:.*?\]', '', content, flags=re.DOTALL)

        # Remove multiple consecutive blank lines
        content = re.sub(r'\n{4,}', '\n\n\n', content)

        # Remove trailing whitespace
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

        return content.strip() + '\n'

    def print_stats(self):
        """Print conversion statistics"""
        print("\n=== Conversion Statistics ===")
        print(f"Math blocks:      {self.stats.math_blocks}")
        print(f"Code blocks:      {self.stats.code_blocks}")
        print(f"Code includes:    {self.stats.code_includes}")
        print(f"Figures:          {self.stats.figures}")
        print(f"Admonitions:      {self.stats.admonitions}")
        print(f"Citations:        {self.stats.citations}")
        print(f"Labels converted: {self.stats.labels_converted}")
        print(f"Refs converted:   {self.stats.refs_converted}")
        print(f"Index removed:    {self.stats.idx_removed}")
        print(f"Headers:          {self.stats.headers_converted}")


def add_yaml_header(content: str, title: str = "") -> str:
    """Add YAML front matter to converted content"""
    yaml = "---\n"
    if title:
        yaml += f'title: "{title}"\n'
    yaml += "---\n\n"
    return yaml + content


def main():
    parser = argparse.ArgumentParser(
        description='Convert DocOnce files to Quarto format'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input DocOnce file')
    parser.add_argument('--output', '-o', required=True,
                        help='Output Quarto file')
    parser.add_argument('--code-base', default='.',
                        help='Base directory for code file resolution')
    parser.add_argument('--no-expand-mako', action='store_true',
                        help='Do not expand Mako variables')
    parser.add_argument('--title', default='',
                        help='Title for YAML header')
    parser.add_argument('--stats', action='store_true',
                        help='Print conversion statistics')

    args = parser.parse_args()

    # Read input file
    with open(args.input, encoding='utf-8') as f:
        content = f.read()

    # Convert
    converter = DocOnceToQuartoConverter(
        code_base=args.code_base,
        expand_mako=not args.no_expand_mako
    )
    converted = converter.convert(content)

    # Add YAML header
    title = args.title or Path(args.input).stem.replace('_', ' ').title()
    converted = add_yaml_header(converted, title)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(converted)

    print(f"Converted: {args.input} -> {args.output}")

    if args.stats:
        converter.print_stats()


if __name__ == '__main__':
    main()
