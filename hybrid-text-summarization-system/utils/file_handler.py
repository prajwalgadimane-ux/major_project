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
    def get_pages_text(pages: list, selected_page_numbers: list) -> str:
        """
        Build the text for only the selected pages, in page order.
        
        Args:
            pages: The page-level structure returned by extract_text() for a PDF,
                   e.g. [{"page_number": 1, "text": "..."}, ...].
            selected_page_numbers: Page numbers to include.
        
        Returns:
            Concatenated text from only the selected pages. Unselected pages'
            text is never included.
        """
        selected_set = set(selected_page_numbers)
        text_parts = []
        for page in pages:
            if page.get("page_number") in selected_set:
                page_text = page.get("text", "")
                if page_text and page_text.strip():
                    text_parts.append(page_text)
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
