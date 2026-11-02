"""
File Handler Module
Handles reading and processing various file formats for text extraction.
Supports: PDF, TXT, DOCX
"""

import os
from typing import Union, BinaryIO
import io


class FileHandler:
    """
    Handles file uploads and text extraction from various formats.
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}
    
    def __init__(self):
        """Initialize the file handler."""
        pass
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension in lowercase."""
        return os.path.splitext(filename)[1].lower()
    
    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if file format is supported."""
        ext = FileHandler.get_file_extension(filename)
        return ext in FileHandler.SUPPORTED_EXTENSIONS
    
    def extract_text(self, file: Union[str, BinaryIO], filename: str = None) -> dict:
        """
        Extract text from a file.
        
        Args:
            file: File path (str) or file-like object
            filename: Original filename (required if file is file-like object)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if hasattr(file, 'seek') and callable(file.seek):
            try:
                file.seek(0)
            except Exception:
                pass

        if isinstance(file, str):
            filename = file
            if not os.path.exists(file):
                return {"success": False, "error": "File not found", "text": ""}
        
        ext = self.get_file_extension(filename)
        
        if ext == '.txt':
            return self._extract_from_txt(file)
        elif ext == '.pdf':
            return self._extract_from_pdf(file)
        elif ext in ['.docx', '.doc']:
            return self._extract_from_docx(file)
        else:
            return {
                "success": False, 
                "error": f"Unsupported format: {ext}",
                "text": ""
            }
    
    def _extract_from_txt(self, file: Union[str, BinaryIO]) -> dict:
        """Extract text from TXT file."""
        try:
            if isinstance(file, str):
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                # Handle file-like object
                content = file.read()
                if isinstance(content, bytes):
                    text = content.decode('utf-8')
                else:
                    text = content
            
            return {
                "success": True,
                "text": text,
                "format": "txt",
                "word_count": len(text.split()),
                "char_count": len(text)
            }
        except Exception as e:
            return {"success": False, "error": str(e), "text": ""}
    
    def _extract_from_pdf(self, file: Union[str, BinaryIO]) -> dict:
        """Extract text from PDF file, preserving per-page structure so callers
        can summarize either the full document or a specific page selection."""
        try:
            import PyPDF2
            
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = self._extract_pdf_pages(reader)
            else:
                # Handle file-like object
                if hasattr(file, 'seek') and callable(file.seek):
                    try:
                        file.seek(0)
                    except Exception:
                        pass
                reader = PyPDF2.PdfReader(file)
                pages = self._extract_pdf_pages(reader)
            
            # Full-document text, built from the page-level structure so both
            # stay perfectly in sync.
            text = '\n'.join(p['text'] for p in pages if p['text'])
            
            return {
                "success": True,
                "text": text,
                "pages": pages,
                "num_pages": len(pages),
                "format": "pdf",
                "word_count": len(text.split()),
                "char_count": len(text)
            }
        except ImportError:
            return {
                "success": False, 
                "error": "PyPDF2 not installed. Run: pip install PyPDF2",
                "text": ""
            }
        except Exception as e:
            return {"success": False, "error": str(e), "text": ""}
    
    def _extract_pdf_pages(self, reader) -> list:
        """Extract text from a PDF reader, preserving page-level structure.
        
        Returns:
            List of dicts, one per page, in page order:
            [{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}, ...]
        """
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            pages.append({"page_number": i, "text": page_text.strip()})
        return pages
    
    @staticmethod
    def parse_page_selection(selection: str, total_pages: int):
        """
        Parse a user-provided page selection string into a sorted list of
        unique, in-range page numbers.
        
        Supports single pages, comma-separated lists, ranges, and combinations:
        "2"   ->  [2]
        "2,5,8"  ->  [2, 5, 8]
        "3-7"  ->  [3, 4, 5, 6, 7]
        "2,4-6,9"  ->  [2, 4, 5, 6, 9]
        
        Args:
            selection: Raw page selection string typed by the user.
            total_pages: Total number of pages in the PDF (for range validation).
        
        Returns:
            Tuple of (page_numbers, error_message). page_numbers is a sorted
            list of ints. error_message is None on success, or a user-facing
            string describing what went wrong.
        """
        if selection is None or not selection.strip():
            return [], "Please enter at least one page number."
        
        if total_pages <= 0:
            return [], "This PDF has no pages to select from."
        
        parts = [p.strip() for p in selection.split(',') if p.strip()]
        if not parts:
            return [], "Please enter at least one page number."
        
        pages = set()
        for part in parts:
            if '-' in part:
                range_parts = part.split('-')
                if len(range_parts) != 2:
                    return [], f"Invalid page range: '{part}'. Use a format like 3-7."
                
                start_str, end_str = range_parts[0].strip(), range_parts[1].strip()
                if not start_str.isdigit() or not end_str.isdigit():
                    return [], f"Invalid page range: '{part}'. Page numbers must be whole numbers."
                
                start, end = int(start_str), int(end_str)
                if start > end:
                    return [], f"Invalid page range: '{part}'. The start page must not be greater than the end page."
                if start < 1 or end > total_pages:
                    return [], f"Page range '{part}' is out of bounds. This PDF only has {total_pages} page(s)."
                
                pages.update(range(start, end + 1))
            else:
                if not part.isdigit():
                    return [], f"Invalid page number: '{part}'. Please enter whole numbers only."
                
                page_num = int(part)
                if page_num < 1 or page_num > total_pages:
                    return [], f"Page {page_num} is out of range. This PDF only has {total_pages} page(s)."
                
                pages.add(page_num)
        
        if not pages:
            return [], "Please enter at least one valid page number."
        
        return sorted(pages), None
    
    @staticmethod
    def identify_repeated_headers_footers(pages: list, min_occurrence: int = 3) -> set:
        """
        Detect running headers and footers that repeat across multiple pages.
        Normalizes lines by stripping numbers/page counters to catch variants like 'Page | 13'.
        """
        import re
        from collections import Counter
        
        normalized_counts = Counter()
        raw_to_norm = {}
        
        for p in pages:
            lines = [l.strip() for l in p.get("text", "").splitlines() if l.strip()]
            if not lines:
                continue
            # Look at candidate header lines (top 4) and footer lines (bottom 4)
            candidates = set(lines[:4] + lines[-4:])
            for line in candidates:
                # Normalize line: remove digits, punctuation, collapse whitespace
                norm = re.sub(r'\d+', '', line).lower()
                norm = re.sub(r'\s+', ' ', norm).strip()
                if len(norm) > 10:  # Only meaningful phrases
                    normalized_counts[norm] += 1
                    raw_to_norm[line] = norm
                    
        threshold = max(min_occurrence, int(len(pages) * 0.25))
        repeated_norms = {norm for norm, count in normalized_counts.items() if count >= threshold}
        
        # Return exact lines that map to repeated patterns
        repeated_lines = {raw for raw, norm in raw_to_norm.items() if norm in repeated_norms}
        return repeated_lines

    @staticmethod
    def clean_text_content(text: str, repeated_lines: set = None, remove_toc_dots: bool = True) -> str:
        """Clean page text by stripping repeated headers/footers, TOC dots, and standalone page numbers."""
        import re
        if not text:
            return ""
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Standalone page numbers or "Page | 12"
            if re.match(r'^(page\s*\|?\s*\d+|\bpage\s+\d+(\s+of\s+\d+)?|\d+)$', stripped, re.IGNORECASE):
                continue
            # TOC dot leaders (e.g., Figure 4.1 ... 27)
            if remove_toc_dots and re.search(r'\.{3,}\s*\d+$', stripped):
                continue
            # Repeated headers/footers
            if repeated_lines and stripped in repeated_lines:
                continue
            if repeated_lines:
                norm = re.sub(r'\d+', '', stripped).lower()
                norm = re.sub(r'\s+', ' ', norm).strip()
                if any(norm == re.sub(r'\d+', '', r).lower().strip() for r in repeated_lines):
                    continue
            cleaned.append(stripped)
        return '\n'.join(cleaned)

    @staticmethod
    def get_pages_dict(pages: list, selected_page_numbers: list, clean_noise: bool = True) -> dict:
        """
        Returns a dictionary mapping {page_num: page_text} for selected pages.
        """
        selected_set = set(selected_page_numbers)
        repeated = FileHandler.identify_repeated_headers_footers(pages) if clean_noise and len(pages) >= 3 else set()
        
        result = {}
        for page in pages:
            num = page.get("page_number")
            if num in selected_set:
                raw_text = page.get("text", "")
                if clean_noise:
                    cleaned = FileHandler.clean_text_content(raw_text, repeated)
                    result[num] = cleaned if cleaned else raw_text
                else:
                    result[num] = raw_text
        return result

    @staticmethod
    def get_pages_text(pages: list, selected_page_numbers: list, clean_noise: bool = True) -> str:
        """
        Build the text for only the selected pages, in page order.
        Strips repetitive headers/footers when clean_noise=True so summarization
        reflects genuine content rather than boilerplate.
        """
        pages_dict = FileHandler.get_pages_dict(pages, selected_page_numbers, clean_noise=clean_noise)
        text_parts = [text for num, text in sorted(pages_dict.items()) if text and text.strip()]
        return '\n\n'.join(text_parts)
    
    @staticmethod
    def format_page_list(page_numbers: list) -> str:
        """Format a list of page numbers into a compact, human-readable string,
        collapsing consecutive runs into ranges, e.g. [2,3,4,7,9,10] -> "2-4, 7, 9-10"."""
        if not page_numbers:
            return ""
        
        nums = sorted(set(page_numbers))
        ranges = []
        start = prev = nums[0]
        
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
                continue
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = n
        
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        return ", ".join(ranges)
    
    def _extract_from_docx(self, file: Union[str, BinaryIO]) -> dict:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            
            if isinstance(file, str):
                doc = Document(file)
            else:
                # Handle file-like object
                doc = Document(file)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text_parts.append(cell.text)
            
            text = '\n'.join(text_parts)
            
            return {
                "success": True,
                "text": text,
                "format": "docx",
                "word_count": len(text.split()),
                "char_count": len(text)
            }
        except ImportError:
            return {
                "success": False, 
                "error": "python-docx not installed. Run: pip install python-docx",
                "text": ""
            }
        except Exception as e:
            return {"success": False, "error": str(e), "text": ""}


def extract_text_from_file(file_path: str) -> str:
    """
    Convenience function to extract text from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text or error message
    """
    handler = FileHandler()
    result = handler.extract_text(file_path)
    
    if result["success"]:
        return result["text"]
    else:
        raise ValueError(result["error"])


def get_supported_formats() -> list:
    """Return list of supported file formats."""
    return list(FileHandler.SUPPORTED_EXTENSIONS)
