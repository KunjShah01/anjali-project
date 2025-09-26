"""
Image generation module for the RAG system.

This module provides image generation capabilities using:
1. Google's Gemini API (gemini-2.5-flash-image-preview) - aka "Nano Banana"
2. Advanced style-specific prompting for various artistic styles
"""

from .generator import (
    GeminiImageGenerator,
    ImageGenerationError,
    create_image_generator,
)

__all__ = ["GeminiImageGenerator", "ImageGenerationError", "create_image_generator"]
