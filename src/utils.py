"""
Utility functions for the Agentic AI application.

This module contains helper functions for text processing, data manipulation,
logging, and other common operations used throughout the application.
"""

import os
import re
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path


# =====================================
# LOGGING UTILITIES
# =====================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration."""
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    return logging.getLogger(__name__)


def log_function_call(func_name: str, args: List[Any], kwargs: Dict[str, Any]) -> None:
    """Log function calls for debugging."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")


# =====================================
# TEXT PROCESSING UTILITIES
# =====================================

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E]', '', text)
    
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to specified length with optional suffix."""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length - len(suffix)]
    return truncated + suffix


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    tokenizer_func: Optional[Callable[[str], int]] = None
) -> List[str]:
    """Split text into chunks based on token count."""
    
    if tokenizer_func is None:
        # Simple approximation: ~4 characters per token
        max_chars = max_tokens * 4
        return split_text_by_chars(text, max_chars)
    
    # Use actual tokenizer if provided
    chunks = []
    current_chunk = ""
    
    sentences = text.split('. ')
    for sentence in sentences:
        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
        
        if tokenizer_func(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def split_text_by_chars(text: str, max_chars: int, overlap: int = 200) -> List[str]:
    """Split text into chunks by character count with overlap."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a sentence or word boundary
        chunk = text[start:end]
        
        # Look for sentence boundary
        last_period = chunk.rfind('. ')
        if last_period > max_chars * 0.5:  # Don't make chunks too small
            end = start + last_period + 2
        else:
            # Look for word boundary
            last_space = chunk.rfind(' ')
            if last_space > max_chars * 0.7:
                end = start + last_space
        
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    
    return chunks


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis."""
    
    # Simple keyword extraction - remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Extract words and count frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return top keywords by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


# =====================================
# DATA MANIPULATION UTILITIES
# =====================================

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely serialize object to JSON with fallback."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return default


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# =====================================
# FILE AND PATH UTILITIES
# =====================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Get MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get file information including size, modification time, etc."""
    path = Path(file_path)
    
    if not path.exists():
        return {}
    
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "extension": path.suffix.lower()
    }


def find_files_by_extension(
    directory: Union[str, Path],
    extensions: List[str],
    recursive: bool = True
) -> List[Path]:
    """Find files by extension in a directory."""
    directory = Path(directory)
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
    
    pattern = "**/*" if recursive else "*"
    
    files = []
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            files.append(file_path)
    
    return files


# =====================================
# TIME AND DATE UTILITIES
# =====================================

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def parse_relative_time(time_str: str) -> datetime:
    """Parse relative time strings like '5 minutes ago', '2 hours ago'."""
    now = datetime.now()
    time_str = time_str.lower().strip()
    
    if time_str in ['now', 'just now']:
        return now
    
    # Extract number and unit
    match = re.match(r'(\d+)\s*(minute|hour|day|week|month)s?\s*ago', time_str)
    if not match:
        return now
    
    amount = int(match.group(1))
    unit = match.group(2)
    
    if unit == 'minute':
        return now - timedelta(minutes=amount)
    elif unit == 'hour':
        return now - timedelta(hours=amount)
    elif unit == 'day':
        return now - timedelta(days=amount)
    elif unit == 'week':
        return now - timedelta(weeks=amount)
    elif unit == 'month':
        return now - timedelta(days=amount * 30)  # Approximate
    
    return now


# =====================================
# MEMORY AND CONTEXT UTILITIES
# =====================================

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap."""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Simple entity extraction using regex patterns."""
    if not text:
        return {}
    
    try:
        entities = {
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            "urls": re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            "phone_numbers": re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
            "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
        }
        
        # Safely create the result dictionary
        result = {}
        for k, v in entities.items():
            if v:  # Only include non-empty lists
                result[k] = list(set(v))
        
        return result
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error extracting entities: {e}")
        return {}


def compress_memory(
    memory_items: List[Dict[str, Any]],
    max_items: int = 100,
    strategy: str = "recent"
) -> List[Dict[str, Any]]:
    """Compress memory by removing older or less relevant items."""
    
    if len(memory_items) <= max_items:
        return memory_items
    
    if strategy == "recent":
        # Keep most recent items
        return memory_items[-max_items:]
    
    elif strategy == "important":
        # Keep items with highest importance score (if available)
        sorted_items = sorted(
            memory_items,
            key=lambda x: x.get("importance", 0),
            reverse=True
        )
        return sorted_items[:max_items]
    
    elif strategy == "diverse":
        # Keep diverse items based on content similarity
        # This is a simplified version - in practice, you might use embeddings
        selected = [memory_items[0]]  # Always keep first item
        
        for item in memory_items[1:]:
            if len(selected) >= max_items:
                break
            
            # Check if item is sufficiently different from selected items
            is_diverse = True
            item_content = item.get("content", "")
            
            for selected_item in selected[-10:]:  # Check against last 10
                selected_content = selected_item.get("content", "")
                similarity = calculate_text_similarity(item_content, selected_content)
                
                if similarity > 0.7:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(item)
        
        return selected
    
    else:
        return memory_items[-max_items:]  # Default to recent


# =====================================
# VALIDATION UTILITIES
# =====================================

def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format."""
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
    return bool(re.match(pattern, url))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


# =====================================
# PERFORMANCE UTILITIES
# =====================================

def time_function(func: Callable) -> Callable:
    """Decorator to time function execution."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(__name__)
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        
        return result
    
    return wrapper


def batch_process(
    items: List[Any],
    process_func: Callable[[List[Any]], List[Any]],
    batch_size: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """Process items in batches with optional progress tracking."""
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i, batch in enumerate(chunk_list(items, batch_size)):
        batch_results = process_func(batch)
        results.extend(batch_results)
        
        if progress_callback:
            progress_callback(i + 1, total_batches)
    
    return results


# =====================================
# ENVIRONMENT UTILITIES
# =====================================

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, "").lower()
    return value in ("true", "1", "yes", "on") if value else default


def get_env_list(key: str, separator: str = ",", default: Optional[List[str]] = None) -> List[str]:
    """Get list from environment variable."""
    if default is None:
        default = []
    
    value = os.getenv(key, "")
    return [item.strip() for item in value.split(separator) if item.strip()] if value else default


# =====================================
# ERROR HANDLING UTILITIES
# =====================================

def safe_execute(func: Callable[[], Any], default: Any = None, log_errors: bool = True) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger = logging.getLogger(__name__)
            logger.error(f"Error executing {func.__name__}: {e}")
        return default


def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: Union[type, Tuple[type, ...]] = Exception
) -> Any:
    """Retry function with exponential backoff."""
    import time
    
    # Ensure exceptions is a tuple
    if not isinstance(exceptions, tuple):
        exceptions = (exceptions,)
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            # Check if the exception type matches any of the expected exceptions
            if not any(isinstance(e, exc_type) for exc_type in exceptions):
                raise e
                
            if attempt == max_retries:
                raise e
            
            wait_time = backoff_factor * (2 ** attempt)
            time.sleep(wait_time)
            
            logger = logging.getLogger(__name__)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")


# =====================================
# SAFE DICTIONARY UTILITIES
# =====================================

def safe_dict_update(target: Dict[str, Any], source: Any) -> Dict[str, Any]:
    """Safely update a dictionary from various sources."""
    if source is None:
        return target
    
    try:
        if isinstance(source, dict):
            target.update(source)
        elif isinstance(source, (list, tuple)):
            # Handle sequences of key-value pairs
            for item in source:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    target[item[0]] = item[1]
                elif isinstance(item, dict):
                    target.update(item)
                else:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Skipping invalid dictionary item: {item}")
        elif hasattr(source, 'items'):
            # Handle objects with items() method
            target.update(source.items())
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"Cannot update dictionary from type: {type(source)}")
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in safe_dict_update: {e}")
    
    return target


def safe_dict_from_sequence(sequence: Any) -> Dict[str, Any]:
    """Safely create a dictionary from a sequence."""
    if not sequence:
        return {}
    
    try:
        if isinstance(sequence, dict):
            return sequence.copy()
        elif isinstance(sequence, (list, tuple)):
            result = {}
            for item in sequence:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    result[item[0]] = item[1]
                elif isinstance(item, dict):
                    result.update(item)
                else:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Skipping invalid sequence item: {item}")
            return result
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"Cannot create dictionary from type: {type(sequence)}")
            return {}
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in safe_dict_from_sequence: {e}")
        return {}


def debug_dict_operation(operation_name: str, data: Any) -> None:
    """Debug dictionary operations to help identify issues."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug(f"Dictionary operation: {operation_name}")
        logger.debug(f"Data type: {type(data)}")
        
        if isinstance(data, (list, tuple)):
            logger.debug(f"Sequence length: {len(data)}")
            for i, item in enumerate(data[:5]):  # Log first 5 items
                logger.debug(f"Item {i}: {type(item)} = {item}")
                if isinstance(item, (list, tuple)):
                    logger.debug(f"  Item length: {len(item)}")
        elif isinstance(data, dict):
            logger.debug(f"Dictionary keys: {list(data.keys())[:10]}")  # First 10 keys
        else:
            logger.debug(f"Data: {str(data)[:100]}")  # First 100 characters
            
    except Exception as e:
        logger.error(f"Error in debug_dict_operation: {e}")


if __name__ == "__main__":
    # Test some utilities
    print("Testing utility functions...")
    
    # Test text processing
    text = "This is a test. It has multiple sentences. And some keywords."
    keywords = extract_keywords(text)
    print(f"Keywords: {keywords}")
    
    # Test text splitting
    chunks = split_text_by_chars(text, 20)
    print(f"Text chunks: {chunks}")
    
    # Test duration formatting
    print(f"Duration: {format_duration(3661)}")
    
    print("Utility tests completed!")
