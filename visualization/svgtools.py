"""SVG manipulation tools for figure composition."""

import xml.etree.ElementTree as ET

# Default figure width in inches
FIG_WIDTH = 6.27

def scale_svg(input_file, output_file, FIG_WIDTH=FIG_WIDTH):
    """Scale SVG figure to specified width while maintaining aspect ratio."""
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Get current dimensions (removing units if present)
    current_width = float(root.get('width').replace('pt', ''))
    current_height = float(root.get('height').replace('pt', ''))

    # Set new dimensions
    new_width = FIG_WIDTH * 72  # in points
    new_height = new_width * (current_height / current_width)

    root.set('width', f'{new_width}pt')
    root.set('height', f'{new_height}pt')

    tree.write(output_file)


def combine_svgs_vertical(file1, file2, output_file):
    """Combine two SVG files vertically, assuming same width."""
    tree1 = ET.parse(file1)
    tree2 = ET.parse(file2)
    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # Get dimensions
    width1 = float(root1.get('width').replace('pt', ''))
    height1 = float(root1.get('height').replace('pt', ''))
    width2 = float(root2.get('width').replace('pt', ''))
    height2 = float(root2.get('height').replace('pt', ''))

    # Get viewBoxes
    viewBox1 = root1.get('viewBox')
    viewBox2 = root2.get('viewBox')

    # Create new SVG
    new_width = width1
    new_height = height1 + height2

    svg_ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', svg_ns)

    new_root = ET.Element('{%s}svg' % svg_ns, {
        'width': f'{new_width}pt',
        'height': f'{new_height}pt',
        'viewBox': f'0 0 {new_width} {new_height}'
    })

    # Add first SVG as nested svg at top
    svg1 = ET.SubElement(new_root, '{%s}svg' % svg_ns, {
        'width': str(width1),
        'height': str(height1),
        'viewBox': viewBox1,
        'x': '0',
        'y': '0'
    })
    for child in root1:
        svg1.append(child)

    # Add second SVG as nested svg below first
    svg2 = ET.SubElement(new_root, '{%s}svg' % svg_ns, {
        'width': str(width2),
        'height': str(height2),
        'viewBox': viewBox2,
        'x': '0',
        'y': str(height1)
    })
    for child in root2:
        svg2.append(child)

    # Write output
    tree = ET.ElementTree(new_root)
    tree.write(output_file, encoding='unicode', xml_declaration=True)


def combine_svgs_horizontal(file1, file2, output_file):
    """Combine two SVG files horizontally."""
    tree1 = ET.parse(file1)
    tree2 = ET.parse(file2)
    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # Get dimensions
    width1 = float(root1.get('width').replace('pt', ''))
    height1 = float(root1.get('height').replace('pt', ''))
    width2 = float(root2.get('width').replace('pt', ''))
    height2 = float(root2.get('height').replace('pt', ''))

    # Get viewBoxes
    viewBox1 = root1.get('viewBox')
    viewBox2 = root2.get('viewBox')

    # Create new SVG
    new_width = width1 + width2
    new_height = max(height1, height2)

    svg_ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', svg_ns)

    new_root = ET.Element('{%s}svg' % svg_ns, {
        'width': f'{new_width}pt',
        'height': f'{new_height}pt',
        'viewBox': f'0 0 {new_width} {new_height}'
    })

    # Add first SVG on left
    svg1 = ET.SubElement(new_root, '{%s}svg' % svg_ns, {
        'width': str(width1),
        'height': str(height1),
        'viewBox': viewBox1,
        'x': '0',
        'y': '0'
    })
    for child in root1:
        svg1.append(child)

    # Add second SVG on right
    svg2 = ET.SubElement(new_root, '{%s}svg' % svg_ns, {
        'width': str(width2),
        'height': str(height2),
        'viewBox': viewBox2,
        'x': str(width1),
        'y': '0'
    })
    for child in root2:
        svg2.append(child)

    # Write output
    tree = ET.ElementTree(new_root)
    tree.write(output_file, encoding='unicode', xml_declaration=True)


def add_text_to_svg(svg_file, output_file, text, x, y, font_family='Arial', font_size=12, font_weight='bold'):
    """Add text to an SVG file at specified coordinates."""
    tree = ET.parse(svg_file)
    root = tree.getroot()

    svg_ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', svg_ns)

    # Add text element
    text_elem = ET.SubElement(root, '{%s}text' % svg_ns, {
        'x': str(x),
        'y': str(y),
        'font-family': font_family,
        'font-size': str(font_size),
        'font-weight': font_weight,
        'fill': 'black'
    })
    text_elem.text = text

    tree.write(output_file, encoding='unicode', xml_declaration=True)


def svg_to_pdf(svg_file, pdf_file):
    """Convert SVG to PDF using cairosvg."""
    # On macOS + Homebrew (Apple Silicon), libcairo lives at /opt/homebrew/lib
    # which is not in the default dyld search path; prepend it so cairocffi's
    # dlopen finds libcairo.2.dylib without requiring a shell-level env var.
    import os, sys
    if sys.platform == 'darwin':
        for candidate in ('/opt/homebrew/lib', '/usr/local/lib'):
            if os.path.isdir(candidate):
                existing = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')
                parts = [p for p in existing.split(':') if p]
                if candidate not in parts:
                    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join([candidate] + parts)
    import cairosvg
    cairosvg.svg2pdf(url=svg_file, write_to=pdf_file)
