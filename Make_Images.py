# import numpy as np
# import cv2
# import os

# # ==============================
# # Adjustable Parameters
# # ==============================
# img_size = 300         # Image dimension (pixels)
# square_size = 100      # Square size (pixels)
# angle_step = 5        # Rotation step (degrees)
# # Define background and square color ranges
# bg_color_range = [(200, 200, 200), (150, 150, 255), (255, 220, 180), (180, 255, 200), (0,0,0)]
# square_color_range = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (200,200,200)]
# # ==============================

# # --- Create output directory ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = os.path.join(current_dir, "clean_images")
# os.makedirs(output_dir, exist_ok=True)

# # --- Loop over each background/square color pair ---
# for i, (bg_color, sq_color) in enumerate(zip(bg_color_range, square_color_range)):

#     # Skip if they are identical (safety)
#     if bg_color == sq_color:
#         continue

#     # --- Create and rotate images for 0–90° ---
#     for angle in range(0, 91, angle_step):

#         # --- Create background ---
#         image = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)

#         # --- Compute square coordinates ---
#         center = img_size // 2
#         half = square_size // 2
#         x1, y1 = center - half, center - half
#         x2, y2 = center + half, center + half

#         # --- Draw square ---
#         cv2.rectangle(image, (x1, y1), (x2, y2), sq_color, -1)

#         # --- Rotate image ---
#         (h, w) = image.shape[:2]
#         center_pt = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
#         rotated = cv2.warpAffine(image, M, (w, h), borderValue=bg_color)

#         # --- Build filename with RGB and angle ---
#         sq_str = f"{sq_color[2]}_{sq_color[1]}_{sq_color[0]}"  # Convert BGR → RGB for readability
#         bg_str = f"{bg_color[2]}_{bg_color[1]}_{bg_color[0]}"
#         filename = f"square_color_{sq_str}_square_angle_{angle}_bckgrnd_{bg_str}.png"

#         # --- Save image ---
#         filepath = os.path.join(output_dir, filename)
#         cv2.imwrite(filepath, rotated)

#         print(f"✅ Saved: {filename}")

# print("\nAll images generated successfully!")


# import numpy as np
# import cv2
# import os

# # ==============================
# # Adjustable Parameters
# # ==============================
# img_size = 300         # Image dimension (pixels)
# triangle_size = 100    # Triangle size (pixels)
# angle_step = 5         # Rotation step (degrees)

# # Define background and triangle color ranges
# bg_color_range = [(200, 200, 200), (150, 150, 255), (255, 220, 180), (180, 255, 200), (0, 0, 0)]
# triangle_color_range = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 200, 200)]
# # ==============================

# # --- Create output directory ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = os.path.join(current_dir, "clean_images")
# os.makedirs(output_dir, exist_ok=True)

# # --- Loop over each background/triangle color pair ---
# for i, (bg_color, tri_color) in enumerate(zip(bg_color_range, triangle_color_range)):

#     # Skip if they are identical (safety)
#     if bg_color == tri_color:
#         continue

#     # --- Create and rotate images for 0–120° ---
#     for angle in range(0, 121, angle_step):

#         # --- Create background ---
#         image = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)

#         # --- Compute triangle coordinates ---
#         center = img_size // 2
#         half = triangle_size // 2

#         # Equilateral triangle points (pointing upward)
#         pts = np.array([
#             [center, center - half],            # top vertex
#             [center - half, center + half],     # bottom left
#             [center + half, center + half]      # bottom right
#         ], np.int32)

#         pts = pts.reshape((-1, 1, 2))

#         # --- Draw triangle ---
#         cv2.fillPoly(image, [pts], tri_color)

#         # --- Rotate image ---
#         (h, w) = image.shape[:2]
#         center_pt = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
#         rotated = cv2.warpAffine(image, M, (w, h), borderValue=bg_color)

#         # --- Build filename with RGB and angle ---
#         tri_str = f"{tri_color[2]}_{tri_color[1]}_{tri_color[0]}"  # Convert BGR → RGB
#         bg_str = f"{bg_color[2]}_{bg_color[1]}_{bg_color[0]}"
#         filename = f"triangle_color_{tri_str}_triangle_angle_{angle}_bckgrnd_{bg_str}.png"

#         # --- Save image ---
#         filepath = os.path.join(output_dir, filename)
#         cv2.imwrite(filepath, rotated)

#         print(f"✅ Saved: {filename}")

# print("\nAll triangle images generated successfully!")


# import numpy as np
# import cv2
# import os

# # ==============================
# # Adjustable Parameters
# # ==============================
# img_size = 400        # Image dimension (pixels)
# tile_size = 100       # Size of each checkered tile
# angle_step = 5        # Rotation step (degrees)
# num_colors = 6        # Number of dynamic color variations
# # ==============================

# # --- Create output directory ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = os.path.join(current_dir, "clean_images")
# os.makedirs(output_dir, exist_ok=True)

# # --- Function to generate dynamic RGB color from loop index ---
# def generate_color(i, total):
#     """Generate a color that smoothly cycles through RGB range."""
#     r = int(127 + 127 * np.sin(2 * np.pi * i / total))
#     g = int(127 + 127 * np.sin(2 * np.pi * (i / total + 1/3)))
#     b = int(127 + 127 * np.sin(2 * np.pi * (i / total + 2/3)))
#     return (b, g, r)  # OpenCV uses BGR

# # --- Loop over color variations ---
# for color_idx in range(num_colors):

#     bg_color = generate_color(color_idx, num_colors)
#     tri_color = generate_color(color_idx + 1, num_colors)  # different phase

#     # --- Create and rotate images for 0–120° ---
#     for angle in range(0, 121, angle_step):

#         # --- Create background ---
#         image = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)

#         # --- Draw checkered pattern made of triangles ---
#         rows = img_size // tile_size
#         cols = img_size // tile_size

#         for r in range(rows):
#             for c in range(cols):
#                 # Alternate colors for checker effect
#                 color = tri_color if (r + c) % 2 == 0 else bg_color

#                 # Coordinates of the tile
#                 x0, y0 = c * tile_size, r * tile_size
#                 x1, y1 = x0 + tile_size, y0 + tile_size

#                 # Draw two triangles per tile (upper-left and lower-right)
#                 pts1 = np.array([[x0, y0], [x1, y0], [x0, y1]], np.int32)  # top-left triangle
#                 pts2 = np.array([[x1, y1], [x0, y1], [x1, y0]], np.int32)  # bottom-right triangle

#                 cv2.fillPoly(image, [pts1], color)
#                 cv2.fillPoly(image, [pts2], color)

#         # --- Rotate entire pattern ---
#         (h, w) = image.shape[:2]
#         center_pt = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
#         rotated = cv2.warpAffine(image, M, (w, h), borderValue=bg_color)

#         # --- Build filename with RGB and angle ---
#         tri_str = f"{tri_color[2]}_{tri_color[1]}_{tri_color[0]}"  # RGB order for readability
#         bg_str = f"{bg_color[2]}_{bg_color[1]}_{bg_color[0]}"
#         filename = f"checkered_triangles_color_{tri_str}_angle_{angle}_bckgrnd_{bg_str}.png"

#         # --- Save image ---
#         filepath = os.path.join(output_dir, filename)
#         cv2.imwrite(filepath, rotated)

#         print(f"✅ Saved: {filename}")

# print("\nAll checkered triangle images generated successfully!")

import numpy as np
import cv2
import os
import random
import argparse

# Lorem ipsum text pool
LOREM_IPSUM = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
    "Duis aute irure dolor in reprehenderit in voluptate velit.",
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa.",
    "Qui officia deserunt mollit anim id est laborum.",
    "Pellentesque habitant morbi tristique senectus et netus.",
    "Vivamus magna justo lacinia eget consectetur sed convallis at.",
    "Curabitur pretium tincidunt lacus nunc pulvinar sapien.",
    "Vestibulum ante ipsum primis in faucibus orci luctus.",
    "Donec sollicitudin molestie malesuada proin libero nunc.",
    "Mauris blandit aliquet elit eget tincidunt nibh pulvinar.",
    "Nulla porttitor accumsan tincidunt cras ultricies ligula sed.",
    "Praesent sapien massa convallis a pellentesque nec egestas.",
    "Quisque velit nisi pretium ut lacinia in elementum id.",
    "Cras adipiscing enim eu turpis egestas pretium aenean.",
    "Nunc sed velit dignissim sodales ut eu sem integer.",
    "Vivamus arcu felis bibendum ut tristique et egestas quis.",
    "Donec adipiscing tristique risus nec feugiat in fermentum.",
    "Lorem ipsum dolor sit amet consectetur adipiscing elit."
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate clean images with Lorem Ipsum text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Generate 50 images in default directory
  %(prog)s
  
  # Generate 100 images with custom size
  %(prog)s -n 100 --width 1024 --height 768
  
  # Specify output directory
  %(prog)s -o my_clean_images -n 30
        """
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="clean_images",
        help="Output directory for generated images (default: clean_images)"
    )
    
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        default=50,
        help="Number of images to generate (default: 50)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Image width in pixels (default: 800)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Image height in pixels (default: 600)"
    )
    
    parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum number of text lines per image (default: 5)"
    )
    
    parser.add_argument(
        "--max-lines",
        type=int,
        default=12,
        help="Maximum number of text lines per image (default: 12)"
    )
    
    return parser.parse_args()

# --- Function to generate random color ---
def random_color():
    """Generate a random BGR color."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# --- Function to get contrasting text color ---
def get_contrast_color(bg_color):
    """Return black or white text depending on background brightness."""
    brightness = (bg_color[2] * 299 + bg_color[1] * 587 + bg_color[0] * 114) / 1000
    return (255, 255, 255) if brightness < 128 else (0, 0, 0)

def main():
    args = parse_args()
    
    # Get parameters from arguments
    img_width = args.width
    img_height = args.height
    num_images = args.num_images
    min_lines = args.min_lines
    max_lines = args.max_lines
    
    # Create output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Clean Text Image Generator")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Image size: {img_width}x{img_height}")
    print(f"Number of images: {num_images}")
    print(f"Text lines per image: {min_lines}-{max_lines}\n")
    
    # Generate images with text
    for img_idx in range(num_images):
        
        # Random background color
        bg_color = random_color()
        text_color = get_contrast_color(bg_color)
        
        # Create background
        image = np.full((img_height, img_width, 3), bg_color, dtype=np.uint8)
        
        # Random font parameters
        font = random.choice([
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX
        ])
        font_scale = random.uniform(0.7, 1.8)
        thickness = random.randint(2, 4)
        line_spacing = random.randint(45, 65)
        
        # Select random lines of text
        num_lines = random.randint(min_lines, max_lines)
        text_lines = random.sample(LOREM_IPSUM, min(num_lines, len(LOREM_IPSUM)))
        
        # Starting position
        y_offset = random.randint(40, 80)
        x_margin = random.randint(30, 60)
        
        # Draw each line of text
        for i, line in enumerate(text_lines):
            y_pos = y_offset + i * line_spacing
            
            # Stop if we're running out of space
            if y_pos > img_height - 30:
                break
            
            # Optionally wrap long lines
            if len(line) > 60:
                
                # Split into two lines
                words = line.split()
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                
                cv2.putText(image, line1, (x_margin, y_pos), font, 
                           font_scale, text_color, thickness, cv2.LINE_AA)
                y_pos += line_spacing
                if y_pos < img_height - 30:
                    cv2.putText(image, line2, (x_margin, y_pos), font, 
                               font_scale, text_color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(image, line, (x_margin, y_pos), font, 
                           font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Build filename
        bg_str = f"{bg_color[2]}_{bg_color[1]}_{bg_color[0]}"
        txt_str = f"{text_color[2]}_{text_color[1]}_{text_color[0]}"
        filename = f"text_image_{img_idx:03d}_bg_{bg_str}_txt_{txt_str}.png"
        
        # Save image
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, image)
        
        print(f"✅ Saved: {filename}")
    
    print(f"\n✅ All {num_images} text images generated successfully!")

if __name__ == "__main__":
    main()
