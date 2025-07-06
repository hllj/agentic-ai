"""
Tools and utilities for LLM interactions.

This module provides various tools that can be used by LLM agents
including search tools, calculation tools, and data processing utilities.
"""

from typing import Any, Dict, List, Optional, Union
import json
import re
import math
import datetime
from dataclasses import dataclass


# =====================================
# TOOL BASE CLASS
# =====================================

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool:
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }


# =====================================
# CALCULATION TOOLS
# =====================================

class CalculatorTool(BaseTool):
    """Simple calculator tool for mathematical operations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic mathematical calculations. Supports +, -, *, /, **, sqrt, and more."
        )
    
    def execute(self, expression: str) -> ToolResult:
        """Execute a mathematical expression safely."""
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Replace common mathematical functions
            expression = expression.replace("sqrt", "math.sqrt")
            expression = expression.replace("sin", "math.sin")
            expression = expression.replace("cos", "math.cos")
            expression = expression.replace("tan", "math.tan")
            expression = expression.replace("log", "math.log")
            expression = expression.replace("exp", "math.exp")
            expression = expression.replace("pi", "math.pi")
            expression = expression.replace("e", "math.e")
            
            # Check for dangerous operations
            dangerous_keywords = ["import", "exec", "eval", "__", "open", "file"]
            if any(keyword in expression.lower() for keyword in dangerous_keywords):
                return ToolResult(
                    success=False,
                    result=None,
                    error="Expression contains potentially dangerous operations"
                )
            
            # Evaluate the expression
            allowed_names = {
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow
            }
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return ToolResult(
                success=True,
                result=result,
                metadata={"expression": expression}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Calculation error: {str(e)}"
            )


class UnitConverterTool(BaseTool):
    """Tool for converting between different units."""
    
    def __init__(self):
        super().__init__(
            name="unit_converter",
            description="Convert between different units (length, weight, temperature, etc.)"
        )
        
        # Conversion factors to base units
        self.conversions = {
            "length": {
                "base": "meter",
                "factors": {
                    "meter": 1.0,
                    "m": 1.0,
                    "kilometer": 1000.0,
                    "km": 1000.0,
                    "centimeter": 0.01,
                    "cm": 0.01,
                    "millimeter": 0.001,
                    "mm": 0.001,
                    "inch": 0.0254,
                    "in": 0.0254,
                    "foot": 0.3048,
                    "ft": 0.3048,
                    "yard": 0.9144,
                    "yd": 0.9144,
                    "mile": 1609.34,
                    "mi": 1609.34
                }
            },
            "weight": {
                "base": "kilogram",
                "factors": {
                    "kilogram": 1.0,
                    "kg": 1.0,
                    "gram": 0.001,
                    "g": 0.001,
                    "pound": 0.453592,
                    "lb": 0.453592,
                    "ounce": 0.0283495,
                    "oz": 0.0283495,
                    "stone": 6.35029
                }
            },
            "temperature": {
                "base": "celsius",
                "special": True  # Special conversion logic needed
            }
        }
    
    def execute(self, value: float, from_unit: str, to_unit: str, unit_type: Optional[str] = None) -> ToolResult:
        """Convert a value from one unit to another."""
        try:
            value = float(value)
            from_unit = from_unit.lower().strip()
            to_unit = to_unit.lower().strip()
            
            # Auto-detect unit type if not provided
            if unit_type is None:
                unit_type = self._detect_unit_type(from_unit, to_unit)
            
            if unit_type not in self.conversions:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Unsupported unit type: {unit_type}"
                )
            
            # Special handling for temperature
            if unit_type == "temperature":
                result = self._convert_temperature(value, from_unit, to_unit)
            else:
                result = self._convert_standard_units(value, from_unit, to_unit, unit_type)
            
            return ToolResult(
                success=True,
                result=result,
                metadata={
                    "original_value": value,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "unit_type": unit_type
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Conversion error: {str(e)}"
            )
    
    def _detect_unit_type(self, from_unit: str, to_unit: str) -> str:
        """Auto-detect the unit type based on the units."""
        for unit_type, data in self.conversions.items():
            if unit_type == "temperature":
                temp_units = ["celsius", "c", "fahrenheit", "f", "kelvin", "k"]
                if from_unit in temp_units or to_unit in temp_units:
                    return "temperature"
            else:
                factors = data.get("factors", {})
                if from_unit in factors and to_unit in factors:
                    return unit_type
        
        return "unknown"
    
    def _convert_standard_units(self, value: float, from_unit: str, to_unit: str, unit_type: str) -> float:
        """Convert between standard units using conversion factors."""
        factors = self.conversions[unit_type]["factors"]
        
        if from_unit not in factors:
            raise ValueError(f"Unknown {unit_type} unit: {from_unit}")
        if to_unit not in factors:
            raise ValueError(f"Unknown {unit_type} unit: {to_unit}")
        
        # Convert to base unit, then to target unit
        base_value = value * factors[from_unit]
        result = base_value / factors[to_unit]
        
        return round(result, 6)
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between temperature units."""
        # Normalize unit names
        unit_map = {
            "c": "celsius", "celsius": "celsius",
            "f": "fahrenheit", "fahrenheit": "fahrenheit", 
            "k": "kelvin", "kelvin": "kelvin"
        }
        
        from_unit = unit_map.get(from_unit, from_unit)
        to_unit = unit_map.get(to_unit, to_unit)
        
        # Convert to Celsius first
        if from_unit == "fahrenheit":
            celsius = (value - 32) * 5/9
        elif from_unit == "kelvin":
            celsius = value - 273.15
        elif from_unit == "celsius":
            celsius = value
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        
        # Convert from Celsius to target unit
        if to_unit == "fahrenheit":
            result = celsius * 9/5 + 32
        elif to_unit == "kelvin":
            result = celsius + 273.15
        elif to_unit == "celsius":
            result = celsius
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
        
        return round(result, 2)


# =====================================
# TEXT PROCESSING TOOLS
# =====================================

class TextAnalyzerTool(BaseTool):
    """Tool for analyzing text properties."""
    
    def __init__(self):
        super().__init__(
            name="text_analyzer",
            description="Analyze text properties like word count, reading time, sentiment, etc."
        )
    
    def execute(self, text: str) -> ToolResult:
        """Analyze the given text."""
        try:
            if not text:
                return ToolResult(
                    success=False,
                    result=None,
                    error="No text provided for analysis"
                )
            
            # Basic text statistics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            paragraphs = text.split('\n\n')
            
            # Character counts
            total_chars = len(text)
            chars_no_spaces = len(text.replace(' ', ''))
            
            # Readability estimates
            avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
            avg_chars_per_word = chars_no_spaces / len(words) if words else 0
            
            # Reading time estimate (average 200 words per minute)
            reading_time_minutes = len(words) / 200
            
            # Simple sentiment analysis (very basic)
            sentiment_score = self._basic_sentiment(text)
            
            analysis = {
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "paragraph_count": len([p for p in paragraphs if p.strip()]),
                "character_count": total_chars,
                "character_count_no_spaces": chars_no_spaces,
                "average_words_per_sentence": round(avg_words_per_sentence, 2),
                "average_characters_per_word": round(avg_chars_per_word, 2),
                "estimated_reading_time_minutes": round(reading_time_minutes, 2),
                "sentiment_score": sentiment_score
            }
            
            return ToolResult(
                success=True,
                result=analysis,
                metadata={"text_length": len(text)}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Text analysis error: {str(e)}"
            )
    
    def _basic_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform basic sentiment analysis."""
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "enjoy", "happy", "pleased", "satisfied", "awesome"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "horrible", "hate", "dislike",
            "angry", "sad", "disappointed", "frustrated", "annoyed", "upset"
        ]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            score = 0.0
        else:
            score = (positive_count - negative_count) / total_sentiment_words
            if score > 0.1:
                sentiment = "positive"
            elif score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": round(score, 3),
            "positive_words": positive_count,
            "negative_words": negative_count
        }


class RegexTool(BaseTool):
    """Tool for regex pattern matching and extraction."""
    
    def __init__(self):
        super().__init__(
            name="regex_tool",
            description="Find patterns in text using regular expressions."
        )
        
        # Common regex patterns
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "time": r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b',
            "ipv4": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "hashtag": r'#\w+',
            "mention": r'@\w+'
        }
    
    def execute(self, text: str, pattern: Optional[str] = None, pattern_name: Optional[str] = None) -> ToolResult:
        """Find patterns in text."""
        try:
            if not text:
                return ToolResult(
                    success=False,
                    result=None,
                    error="No text provided for pattern matching"
                )
            
            # Determine which pattern to use
            if pattern_name and pattern_name in self.patterns:
                regex_pattern = self.patterns[pattern_name]
                used_pattern = f"{pattern_name}: {regex_pattern}"
            elif pattern:
                regex_pattern = pattern
                used_pattern = f"custom: {pattern}"
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error="No pattern or pattern_name provided"
                )
            
            # Find all matches
            matches = re.findall(regex_pattern, text, re.IGNORECASE)
            
            # Find matches with positions
            match_objects = list(re.finditer(regex_pattern, text, re.IGNORECASE))
            detailed_matches = []
            
            for match in match_objects:
                detailed_matches.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "groups": match.groups() if match.groups() else []
                })
            
            result = {
                "pattern_used": used_pattern,
                "matches": matches,
                "match_count": len(matches),
                "detailed_matches": detailed_matches,
                "available_patterns": list(self.patterns.keys())
            }
            
            return ToolResult(
                success=True,
                result=result,
                metadata={"text_length": len(text)}
            )
            
        except re.error as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Invalid regex pattern: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Regex tool error: {str(e)}"
            )


# =====================================
# DATE AND TIME TOOLS
# =====================================

class DateTimeTool(BaseTool):
    """Tool for date and time operations."""
    
    def __init__(self):
        super().__init__(
            name="datetime_tool",
            description="Perform date and time calculations and formatting."
        )
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Perform date/time operations."""
        try:
            if operation == "current":
                return self._get_current_datetime(**kwargs)
            elif operation == "format":
                return self._format_datetime(**kwargs)
            elif operation == "add":
                return self._add_time(**kwargs)
            elif operation == "subtract":
                return self._subtract_time(**kwargs)
            elif operation == "difference":
                return self._calculate_difference(**kwargs)
            elif operation == "parse":
                return self._parse_datetime(**kwargs)
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Unknown operation: {operation}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"DateTime tool error: {str(e)}"
            )
    
    def _get_current_datetime(self, timezone: Optional[str] = None, format_string: Optional[str] = None) -> ToolResult:
        """Get current date and time."""
        now = datetime.datetime.now()
        
        result = {
            "datetime": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.time().isoformat(),
            "timestamp": now.timestamp(),
            "formatted": now.strftime(format_string) if format_string else str(now)
        }
        
        return ToolResult(success=True, result=result)
    
    def _format_datetime(self, dt_string: str, format_string: str) -> ToolResult:
        """Format a datetime string."""
        dt = datetime.datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        formatted = dt.strftime(format_string)
        
        return ToolResult(
            success=True,
            result={"formatted": formatted, "original": dt_string}
        )
    
    def _add_time(self, dt_string: str, **time_delta) -> ToolResult:
        """Add time to a datetime."""
        dt = datetime.datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        delta = datetime.timedelta(**time_delta)
        new_dt = dt + delta
        
        return ToolResult(
            success=True,
            result={
                "original": dt.isoformat(),
                "new_datetime": new_dt.isoformat(),
                "added": str(delta)
            }
        )
    
    def _subtract_time(self, dt_string: str, **time_delta) -> ToolResult:
        """Subtract time from a datetime."""
        dt = datetime.datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        delta = datetime.timedelta(**time_delta)
        new_dt = dt - delta
        
        return ToolResult(
            success=True,
            result={
                "original": dt.isoformat(),
                "new_datetime": new_dt.isoformat(),
                "subtracted": str(delta)
            }
        )
    
    def _calculate_difference(self, dt1_string: str, dt2_string: str) -> ToolResult:
        """Calculate difference between two datetimes."""
        dt1 = datetime.datetime.fromisoformat(dt1_string.replace('Z', '+00:00'))
        dt2 = datetime.datetime.fromisoformat(dt2_string.replace('Z', '+00:00'))
        
        difference = dt2 - dt1
        
        return ToolResult(
            success=True,
            result={
                "datetime1": dt1.isoformat(),
                "datetime2": dt2.isoformat(),
                "difference": str(difference),
                "total_seconds": difference.total_seconds(),
                "days": difference.days
            }
        )
    
    def _parse_datetime(self, dt_string: str, format_string: Optional[str] = None) -> ToolResult:
        """Parse a datetime string."""
        if format_string:
            dt = datetime.datetime.strptime(dt_string, format_string)
        else:
            # Try common formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ"
            ]
            
            dt = None
            for fmt in formats:
                try:
                    dt = datetime.datetime.strptime(dt_string, fmt)
                    break
                except ValueError:
                    continue
            
            if dt is None:
                raise ValueError(f"Could not parse datetime: {dt_string}")
        
        return ToolResult(
            success=True,
            result={
                "original": dt_string,
                "parsed": dt.isoformat(),
                "date": dt.date().isoformat(),
                "time": dt.time().isoformat()
            }
        )


# =====================================
# TOOL MANAGER
# =====================================

class ToolManager:
    """Manager for organizing and executing tools."""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        self.register_tool(CalculatorTool())
        self.register_tool(UnitConverterTool())
        self.register_tool(TextAnalyzerTool())
        self.register_tool(RegexTool())
        self.register_tool(DateTimeTool())
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        return tool.execute(*args, **kwargs)


# Create global tool manager instance
tool_manager = ToolManager()


if __name__ == "__main__":
    # Test tools
    print("Testing tools...")
    
    # Test calculator
    calc_result = tool_manager.execute_tool("calculator", "2 + 3 * 4")
    print(f"Calculator result: {calc_result}")
    
    # Test unit converter
    conv_result = tool_manager.execute_tool("unit_converter", 100, "fahrenheit", "celsius")
    print(f"Unit conversion result: {conv_result}")
    
    # Test text analyzer
    text_result = tool_manager.execute_tool("text_analyzer", "Hello world! This is a test.")
    print(f"Text analysis result: {text_result}")
    
    # List all tools
    print(f"Available tools: {[tool['name'] for tool in tool_manager.list_tools()]}")
    
    print("Tool testing completed!")
