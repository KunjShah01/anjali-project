"""
Image generation service using Google's Gemini API for "Nano Banana" image generation.
"""

import base64
import io
from typing import Dict, Any, Optional
from datetime import datetime
from google import genai
from google.genai import types
from PIL import Image
import json

from ..config import GeminiConfig
from ..utils.logger import LoggerMixin


class ImageGenerationError(Exception):
    """Exception raised when image generation fails."""

    pass


class GeminiImageGenerator(LoggerMixin):
    """
    Image generation service using Google's official Gemini API for "Nano Banana" generation.
    """

    def __init__(self, config: GeminiConfig):
        super().__init__()
        self.config = config
        self.client = None
        self._configure_client()

        # Model for image generation
        self.image_model = "gemini-2.5-flash-image-preview"

        # Statistics
        self.stats = {
            "images_generated": 0,
            "total_generation_time": 0.0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Simple cache for generated images
        self._image_cache: Dict[str, Dict[str, Any]] = {}

    def _configure_client(self):
        """Configure Google Generative AI client."""
        if not self.config.api_key:
            raise ValueError("Gemini API key is required for image generation")

        try:
            self.client = genai.Client(api_key=self.config.api_key)
            self.log_info("Gemini client configured successfully")
        except Exception as e:
            self.log_error("Failed to configure Gemini client", error=e)
            raise

    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for the request."""
        cache_data = {"prompt": prompt, **kwargs}
        return str(hash(json.dumps(cache_data, sort_keys=True)))

    async def generate_image_with_gemini(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        style: str = "photorealistic",
        quality: str = "standard",
    ) -> Dict[str, Any]:
        """
        Generate image using Google's official Gemini API (aka "Nano Banana").

        Args:
            prompt: Text prompt for image generation
            width: Image width in pixels
            height: Image height in pixels
            style: Generation style (photorealistic, artistic, etc.)
            quality: Image quality (standard, high)

        Returns:
            Dictionary with image data and metadata
        """
        if not self.client:
            raise ImageGenerationError("Gemini client not configured")

        start_time = datetime.now()

        try:
            # Check cache first
            cache_key = self._get_cache_key(
                prompt, width=width, height=height, style=style
            )
            if cache_key in self._image_cache:
                self.stats["cache_hits"] += 1
                self.log_info("Retrieved image from cache", prompt=prompt[:50])
                return self._image_cache[cache_key]

            self.stats["cache_misses"] += 1

            # Enhanced prompt with style and quality specifications
            enhanced_prompt = self._enhance_prompt(
                prompt, style, quality, width, height
            )

            self.log_info("Generating image with Gemini API", prompt=prompt[:50])

            # Use the official Google Gemini API
            response = self.client.models.generate_content(
                model=self.image_model,
                contents=[enhanced_prompt],
            )

            # Process the response
            image_data = None
            text_response = None

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    text_response = part.text
                    self.log_info("Received text response", text=text_response[:100])
                elif part.inline_data is not None:
                    # Convert the inline data to base64
                    image_data = base64.b64encode(part.inline_data.data).decode()
                    break

            if not image_data:
                raise ImageGenerationError("No image data received from Gemini API")

            # Create result
            result = {
                "image_data": image_data,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "width": width,
                "height": height,
                "style": style,
                "quality": quality,
                "generation_time": (datetime.now() - start_time).total_seconds(),
                "model": self.image_model,
                "text_response": text_response,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache the result
            self._image_cache[cache_key] = result

            # Update statistics
            self.stats["images_generated"] += 1
            self.stats["total_generation_time"] += result["generation_time"]

            self.log_info(
                "Image generated successfully with Gemini API",
                prompt=prompt[:50],
                generation_time=result["generation_time"],
            )

            return result

        except Exception as e:
            self.stats["errors"] += 1
            self.log_error(
                "Gemini API image generation failed", error=e, prompt=prompt[:50]
            )
            raise ImageGenerationError(f"Gemini API error: {str(e)}")

    def _enhance_prompt(
        self, prompt: str, style: str, quality: str, width: int, height: int
    ) -> str:
        """
        Enhance the prompt with style, quality, and dimension specifications.
        """
        # Style mappings for better results
        style_enhancements = {
            "photorealistic": "photorealistic, high-resolution, professional photography",
            "artistic": "artistic, creative, expressive, painterly",
            "abstract": "abstract art, non-representational, creative interpretation",
            "cinematic": "cinematic lighting, movie-like composition, dramatic",
            "anime": "anime style, Japanese animation, colorful, expressive",
            "cartoon": "cartoon style, animated, colorful, playful",
            "fantasy": "fantasy art, magical, ethereal, imaginative",
            "cyberpunk": "cyberpunk aesthetic, neon lights, futuristic, digital",
        }

        # Quality specifications
        quality_specs = {
            "standard": "clear, well-composed",
            "high": "ultra-high quality, sharp details, professional grade",
        }

        # Build enhanced prompt
        style_text = style_enhancements.get(style, style)
        quality_text = quality_specs.get(quality, quality)

        enhanced = f"{prompt}. {style_text}, {quality_text}"

        # Add dimension hint if needed for composition
        if width > height:
            enhanced += ", landscape orientation"
        elif height > width:
            enhanced += ", portrait orientation"
        else:
            enhanced += ", square composition"

        return enhanced

    async def generate_image_with_nano_banana(
        self, prompt: str, style: str = "anime", steps: int = 20, cfg_scale: float = 7.0
    ) -> Dict[str, Any]:
        """
        Generate image using Nano Banana style (this is actually Gemini's nickname).
        This method provides style-specific enhancements for anime and artistic styles.

        Args:
            prompt: Text prompt for image generation
            style: Art style (anime, realistic, cartoon, etc.)
            steps: Number of inference steps (ignored for Gemini, kept for compatibility)
            cfg_scale: CFG scale for prompt adherence (ignored for Gemini)

        Returns:
            Dictionary with image data and metadata
        """
        if not self.client:
            raise ImageGenerationError("Gemini client not configured")

        start_time = datetime.now()

        try:
            # Check cache
            cache_key = self._get_cache_key(
                prompt, style=style, steps=steps, cfg_scale=cfg_scale
            )
            if cache_key in self._image_cache:
                self.stats["cache_hits"] += 1
                self.log_info(
                    "Retrieved Nano Banana image from cache", prompt=prompt[:50]
                )
                return self._image_cache[cache_key]

            self.stats["cache_misses"] += 1

            # Create style-specific enhanced prompt
            enhanced_prompt = self._create_style_prompt(prompt, style)

            self.log_info(
                "Generating Nano Banana style image", prompt=prompt[:50], style=style
            )

            # Use the Gemini API with style-specific prompting
            response = self.client.models.generate_content(
                model=self.image_model,
                contents=[enhanced_prompt],
            )

            # Process the response
            image_data = None
            text_response = None

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    text_response = part.text
                elif part.inline_data is not None:
                    image_data = base64.b64encode(part.inline_data.data).decode()
                    break

            if not image_data:
                raise ImageGenerationError(
                    "No image data received from Nano Banana generation"
                )

            # Create result
            result = {
                "image_data": image_data,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "style": style,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "generation_time": (datetime.now() - start_time).total_seconds(),
                "provider": "nano_banana",
                "model": self.image_model,
                "text_response": text_response,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache and update stats
            self._image_cache[cache_key] = result
            self.stats["images_generated"] += 1
            self.stats["total_generation_time"] += result["generation_time"]

            self.log_info(
                "Nano Banana image generated successfully",
                prompt=prompt[:50],
                generation_time=result["generation_time"],
            )

            return result

        except Exception as e:
            self.stats["errors"] += 1
            self.log_error(
                "Nano Banana image generation failed", error=e, prompt=prompt[:50]
            )
            raise ImageGenerationError(f"Nano Banana generation error: {str(e)}")

    def _create_style_prompt(self, prompt: str, style: str) -> str:
        """
        Create style-specific prompts for different artistic styles.
        """
        style_prompts = {
            "anime": f"Create an anime-style illustration: {prompt}. Japanese animation style, vibrant colors, expressive characters, detailed artwork, clean lines.",
            "realistic": f"Create a photorealistic image: {prompt}. High-resolution photography, natural lighting, realistic textures and materials.",
            "cartoon": f"Create a cartoon-style illustration: {prompt}. Animated style, bright colors, simplified forms, playful and fun.",
            "fantasy": f"Create a fantasy art piece: {prompt}. Magical, mystical atmosphere, rich colors, detailed fantasy elements, epic composition.",
            "cyberpunk": f"Create a cyberpunk-style image: {prompt}. Futuristic, neon lighting, high-tech aesthetic, urban dystopian atmosphere.",
            "watercolor": f"Create a watercolor painting: {prompt}. Soft brushstrokes, flowing colors, artistic medium, painterly effect.",
            "oil_painting": f"Create an oil painting: {prompt}. Rich textures, classical painting technique, artistic brushwork, traditional art style.",
            "sketch": f"Create a pencil sketch: {prompt}. Hand-drawn appearance, graphite lines, artistic sketching style, monochromatic.",
        }

        return style_prompts.get(
            style,
            f"Create a {style} style image: {prompt}. Artistic, creative, and visually appealing.",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get image generation statistics."""
        avg_time = (
            self.stats["total_generation_time"] / self.stats["images_generated"]
            if self.stats["images_generated"] > 0
            else 0
        )

        return {
            **self.stats,
            "average_generation_time": avg_time,
            "cache_size": len(self._image_cache),
            "supported_providers": ["gemini_imagen", "nano_banana"],
        }

    def clear_cache(self):
        """Clear the image cache."""
        self._image_cache.clear()
        self.log_info("Image cache cleared")

    def save_image(self, image_data: str, filename: str, format: str = "PNG") -> str:
        """
        Save base64 image data to file.

        Args:
            image_data: Base64 encoded image data
            filename: Output filename
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Path to saved file
        """
        try:
            # Decode base64 data
            image_bytes = base64.b64decode(image_data)

            # Create PIL image
            image = Image.open(io.BytesIO(image_bytes))

            # Save image
            image.save(filename, format=format)

            self.log_info("Image saved successfully", filename=filename)
            return filename

        except Exception as e:
            self.log_error("Failed to save image", error=e, filename=filename)
            raise


def create_image_generator(
    config: Optional[GeminiConfig] = None,
) -> GeminiImageGenerator:
    """
    Factory function to create image generator.

    Args:
        config: Optional GeminiConfig. If None, uses default config.

    Returns:
        GeminiImageGenerator instance
    """
    if config is None:
        from ..config import get_config

        config = get_config().gemini

    return GeminiImageGenerator(config)
