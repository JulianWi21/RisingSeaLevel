"""
Simple demo script to create a visualization of rising sea levels.
Uses synthetic terrain data - no downloads required.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse


def generate_synthetic_terrain(width, height, seed=42):
    """Generate a simple synthetic terrain using Perlin-like noise."""
    np.random.seed(seed)
    
    # Create base elevation with multiple octaves of noise
    terrain = np.zeros((height, width))
    
    # Add different frequency noise layers
    for octave in range(5):
        freq = 2 ** octave
        amplitude = 100 / (2 ** octave)
        
        # Simple smoothed random noise
        noise = np.random.randn(height // freq + 2, width // freq + 2)
        
        # Interpolate to full size
        from scipy.ndimage import zoom
        noise_full = zoom(noise, freq, order=1)
        
        # Crop to exact size
        noise_full = noise_full[:height, :width]
        terrain += noise_full * amplitude
    
    # Make it more realistic - add a gradient (lower at edges)
    y_coords, x_coords = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing='ij')
    distance_from_center = np.sqrt(x_coords**2 + y_coords**2)
    edge_factor = 1.0 - np.clip(distance_from_center / 1.4, 0, 1)
    
    terrain = terrain * edge_factor + 50
    
    # Shift so sea level 0 is meaningful
    terrain = terrain - terrain.min() + 10
    
    return terrain


def terrain_to_color(elevation, sea_level=0):
    """Convert elevation to RGB color, accounting for sea level."""
    # Define color gradient for terrain
    color_stops = [
        (0,    np.array([30, 120, 50])),    # Low green
        (50,   np.array([60, 160, 60])),    # Green
        (100,  np.array([100, 180, 70])),   # Light green
        (200,  np.array([160, 200, 80])),   # Yellow-green
        (300,  np.array([200, 210, 100])),  # Yellow
        (400,  np.array([220, 200, 80])),   # Brown-yellow
        (500,  np.array([220, 170, 60])),   # Brown
    ]
    
    # Ocean colors
    ocean_deep = np.array([40, 80, 160])
    ocean_shore = np.array([100, 160, 220])
    
    h, w = elevation.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Determine what's under water
    underwater = elevation <= sea_level
    
    # Color land
    for i in range(len(color_stops) - 1):
        h_low, c_low = color_stops[i]
        h_high, c_high = color_stops[i + 1]
        
        # Interpolate colors
        mask = (elevation > h_low) & (elevation <= h_high) & ~underwater
        if np.any(mask):
            t = (elevation[mask] - h_low) / (h_high - h_low)
            for ch in range(3):
                rgb[mask, ch] = (c_low[ch] * (1 - t) + c_high[ch] * t).astype(np.uint8)
    
    # Handle extremes
    very_low = (elevation <= color_stops[0][0]) & ~underwater
    very_high = (elevation > color_stops[-1][0]) & ~underwater
    rgb[very_low] = color_stops[0][1]
    rgb[very_high] = color_stops[-1][1]
    
    # Color ocean
    if np.any(underwater):
        # Depth-based color
        depth = sea_level - elevation[underwater]
        depth_normalized = np.clip(depth / 50, 0, 1)  # Normalize to 0-1
        
        for ch in range(3):
            ocean_color = (ocean_shore[ch] * (1 - depth_normalized) + 
                          ocean_deep[ch] * depth_normalized)
            rgb[underwater, ch] = ocean_color.astype(np.uint8)
    
    return rgb


def create_image_with_overlay(terrain, sea_level, width, height, title):
    """Create the final image with text overlay."""
    # Render terrain
    rgb = terrain_to_color(terrain, sea_level)
    
    # Create PIL image
    img = Image.fromarray(rgb)
    
    # Resize if needed
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    # Add text overlay
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw title
    text_color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    
    # Title with shadow
    title_pos = (30, 30)
    draw.text((title_pos[0] + 2, title_pos[1] + 2), title, fill=shadow_color, font=font_large)
    draw.text(title_pos, title, fill=text_color, font=font_large)
    
    # Sea level indicator
    sea_text = f"Sea Level: +{sea_level:.0f}m"
    sea_pos = (30, height - 70)
    draw.text((sea_pos[0] + 2, sea_pos[1] + 2), sea_text, fill=shadow_color, font=font_small)
    draw.text(sea_pos, sea_text, fill=text_color, font=font_small)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Create a demo visualization of rising sea levels")
    parser.add_argument("--output", default="demo_sea_level_rise.png", 
                       help="Output image filename (default: demo_sea_level_rise.png)")
    parser.add_argument("--width", type=int, default=1920,
                       help="Image width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                       help="Image height (default: 1080)")
    parser.add_argument("--sea-level", type=float, default=50,
                       help="Sea level in meters (default: 50)")
    parser.add_argument("--terrain-size", type=int, default=800,
                       help="Terrain resolution (default: 800)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for terrain generation (default: 42)")
    
    args = parser.parse_args()
    
    print(f"Generating synthetic terrain ({args.terrain_size}x{args.terrain_size})...")
    terrain = generate_synthetic_terrain(args.terrain_size, args.terrain_size, args.seed)
    
    print(f"Creating image at {args.width}x{args.height} with sea level +{args.sea_level}m...")
    title = "Rising Sea Level Visualization"
    img = create_image_with_overlay(terrain, args.sea_level, args.width, args.height, title)
    
    print(f"Saving to {args.output}...")
    img.save(args.output, quality=95)
    
    print(f"✓ Demo image created successfully!")
    print(f"  - File: {args.output}")
    print(f"  - Size: {args.width}x{args.height}")
    print(f"  - Sea level: +{args.sea_level}m")


if __name__ == "__main__":
    main()
