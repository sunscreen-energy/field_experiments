#!/usr/bin/env python3
"""
Convert markdown files to PDF with embedded images.

Usage:
    python markdown_to_pdf.py <input.md> [output.pdf]

If output.pdf is not specified, it will use the same name as the input file.
"""

import sys
import markdown
from pathlib import Path
from xhtml2pdf import pisa


def markdown_to_pdf(markdown_path, pdf_path=None):
    """
    Convert a markdown file to PDF with embedded images.

    Args:
        markdown_path: Path to the markdown file
        pdf_path: Path to output PDF file (optional)
    """
    markdown_path = Path(markdown_path)

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    if pdf_path is None:
        pdf_path = markdown_path.with_suffix('.pdf')
    else:
        pdf_path = Path(pdf_path)

    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    md = markdown.Markdown(extensions=['extra', 'tables', 'fenced_code'])
    html_content = md.convert(markdown_text)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 2em;
                color: #24292e;
            }}
            h1, h2, h3, h4, h5, h6 {{
                margin-top: 24px;
                margin-bottom: 16px;
                font-weight: 600;
                line-height: 1.25;
            }}
            h1 {{ font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
            h2 {{ font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
            h3 {{ font-size: 1.25em; }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 1em 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1em 0;
            }}
            table th, table td {{
                border: 1px solid #dfe2e5;
                padding: 6px 13px;
            }}
            table th {{
                background-color: #f6f8fa;
                font-weight: 600;
            }}
            table tr:nth-child(2n) {{
                background-color: #f6f8fa;
            }}
            blockquote {{
                border-left: 4px solid #dfe2e5;
                padding-left: 1em;
                margin-left: 0;
                color: #6a737d;
            }}
            code {{
                background-color: rgba(27,31,35,0.05);
                border-radius: 3px;
                padding: 0.2em 0.4em;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 85%;
            }}
            pre {{
                background-color: #f6f8fa;
                border-radius: 3px;
                padding: 16px;
                overflow: auto;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            hr {{
                border: 0;
                border-top: 2px solid #eaecef;
                margin: 24px 0;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    with open(pdf_path, 'wb') as pdf_file:
        pisa_status = pisa.CreatePDF(
            html_template,
            dest=pdf_file,
            path=str(markdown_path.parent.absolute())
        )

    if pisa_status.err:
        raise RuntimeError(f"PDF generation failed with errors")

    print(f"PDF created: {pdf_path}")
    return pdf_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    markdown_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    markdown_to_pdf(markdown_file, output_file)
