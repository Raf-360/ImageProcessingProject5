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
# output_dir = os.path.join(current_dir, "output_images")
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
# output_dir = os.path.join(current_dir, "output_images")
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
# output_dir = os.path.join(current_dir, "output_images")
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

# ==============================
# Adjustable Parameters
# ==============================
img_size = 400        # Image dimension (pixels)
triangle_size = 100   # Side length of each triangle
angle_step = 5        # Rotation step (degrees)
num_colors = 6        # Number of dynamic color variations
# ==============================

# --- Create output directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output_triangles")
os.makedirs(output_dir, exist_ok=True)

# --- Function to generate dynamic color ---
def generate_color(i, total):
    """Generate a smooth, cyclic RGB color."""
    r = int(127 + 127 * np.sin(2 * np.pi * i / total))
    g = int(127 + 127 * np.sin(2 * np.pi * (i / total + 1/3)))
    b = int(127 + 127 * np.sin(2 * np.pi * (i / total + 2/3)))
    return (b, g, r)  # OpenCV uses BGR

# --- Loop through color variations ---
for color_idx in range(num_colors):

    color1 = generate_color(color_idx, num_colors)
    color2 = generate_color(color_idx + 1, num_colors)

    # --- Loop through rotation angles ---
    for angle in range(0, 121, angle_step):

        # Background base
        image = np.full((img_size, img_size, 3), color1, dtype=np.uint8)

        # Height of equilateral triangle
        h_tri = int(np.sqrt(3) / 2 * triangle_size)
        rows = img_size // h_tri + 1
        cols = img_size // triangle_size + 1

        # Draw triangle pattern
        for r in range(rows):
            for c in range(cols):
                # Determine color alternation
                tri_color = color2 if (r + c) % 2 == 0 else color1

                # Shift alternate rows
                x_offset = (triangle_size // 2) if r % 2 else 0

                # Triangle vertex base coordinates
                x0 = c * triangle_size + x_offset
                y0 = r * h_tri

                # Upward or downward triangle
                if (r + c) % 2 == 0:
                    pts = np.array([
                        [x0, y0 + h_tri],
                        [x0 + triangle_size / 2, y0],
                        [x0 + triangle_size, y0 + h_tri]
                    ], np.int32)
                else:
                    pts = np.array([
                        [x0, y0],
                        [x0 + triangle_size / 2, y0 + h_tri],
                        [x0 + triangle_size, y0]
                    ], np.int32)

                cv2.fillPoly(image, [pts], tri_color)

        # --- Rotate pattern ---
        (h, w) = image.shape[:2]
        center_pt = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=color1)

        # --- Save file ---
        c1_str = f"{color1[2]}_{color1[1]}_{color1[0]}"
        c2_str = f"{color2[2]}_{color2[1]}_{color2[0]}"
        filename = f"triangle_checker_color_{c2_str}_angle_{angle}_bckgrnd_{c1_str}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, rotated)

        print(f"✅ Saved: {filename}")

print("\nAll true triangle checkerboard images generated successfully!")
