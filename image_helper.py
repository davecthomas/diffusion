import csv
import glob
import os
import re
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from datetime import datetime


class ImageHelper:
    def __init__(self, image_folder="images", logo_folder="logo"):
        # Timestamped log file
        self.csv_file = f"image_log_{
            datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.image_folder = image_folder
        self.logo_folder = logo_folder
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.logo_folder, exist_ok=True)

    def sanitize_filename(self, text: str) -> str:
        """
        Sanitizes a string to make it a valid filename by removing or replacing invalid characters.
        """
        return re.sub(r'[^\w\-_\.]', '_', text).strip()[:96]

    def save_raw_image(self, img_data: bytes, prompt: str, idx: int) -> str:
        """
        Saves the raw image data to a file and returns the filename.

        Args:
            img_data (bytes): The raw image data to save.
            prompt (str): The prompt used to generate the image.
            idx (int): The index of the image (in case of multiple images for the same prompt).

        Returns:
            str: The filename where the image was saved.
        """
        sanitized_prompt = self.sanitize_filename(prompt)
        raw_image_filename = f"{
            self.image_folder}/{sanitized_prompt}_{idx + 1}_raw.png"
        with open(raw_image_filename, 'wb') as raw_img_file:
            raw_img_file.write(img_data)
        return raw_image_filename

    def crop_image(self, image_path: str, target_width: int, target_height: int) -> Image.Image:
        """
        Opens the image from the specified path, crops it to the target dimensions, and returns the cropped image.
        """
        try:
            with Image.open(image_path) as img:
                cropped_img = ImageOps.fit(
                    img, (target_width, target_height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            return cropped_img
        except Exception as e:
            print(f"Error cropping image {image_path}: {e}")
            raise

    def save_image(self, img: Image.Image, filename: str):
        """
        Saves the image to the images/ directory with a date and timestamp prepended to the filename.
        """
        # Get current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Prepend the timestamp to the filename
        filename_with_timestamp = f"{timestamp}_{filename}"

        # Construct the full path for the file
        final_filename = os.path.join(
            self.image_folder, filename_with_timestamp)

        # Save the image with the timestamped filename
        img.save(final_filename)
        print(f"Image saved as {final_filename}")

    def log_to_csv(self, prompt: str, dimensions: tuple, filename: str):
        """
        Logs the prompt, dimensions, and filename to a CSV file with a header row.
        """
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Prompt", "Width", "Height", "Filename"])
            writer.writerow([prompt, dimensions[0], dimensions[1], filename])
        print(f"Logged prompt and details to {self.csv_file}")

    def cleanup_raw_files(self):
        """
        Deletes all files that contain 'raw.png' in their filename.
        """
        raw_file_pattern = os.path.join(self.image_folder, "*raw.png")
        files_to_delete = glob.glob(raw_file_pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    def add_logo_to_image(self, image: Image.Image) -> Image.Image:
        """
        Loads the first logo file from the logo folder, places it at the top center of the image, 
        resizes the logo to fit comfortably, adds a glow effect, and applies a border to the logo 
        for visibility on both light and dark backgrounds.

        Args:
            image (Image.Image): The image on which the logo will be placed.

        Returns:
            Image.Image: The modified image with the logo added.
        """
        try:
            # Find the first logo file in the logo folder
            logo_files = glob.glob(os.path.join(self.logo_folder, "*"))
            if not logo_files:
                print("No logo files found. Returning the original image.")
                return image

            logo_path = logo_files[0]
            logo = Image.open(logo_path)

            # Resize the logo to fit comfortably at the top of the image (50% of image width)
            # Set the logo width to 50% of the main image width
            max_logo_width = image.width // 2
            logo_ratio = logo.width / logo.height
            logo_height = int(max_logo_width / logo_ratio)
            logo = logo.resize((max_logo_width, logo_height),
                               Image.Resampling.LANCZOS)

            # Add a white or black border depending on the background
            border_color = (255, 255, 255)  # White border for dark backgrounds
            logo_with_border = ImageOps.expand(
                logo, border=10, fill=border_color)

            # Create a glow effect for the logo
            # Apply a Gaussian blur for the glow effect
            glow = logo_with_border.filter(ImageFilter.GaussianBlur(15))

            # Create a new image for the logo with a transparent background
            glow_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
            logo_img = Image.new("RGBA", image.size, (0, 0, 0, 0))

            # Calculate the position to center the logo at the top
            logo_x = (image.width - logo_with_border.width) // 2
            logo_y = 20  # Add some spacing from the top of the image

            # Paste the glow first, then the logo on top
            glow_img.paste(glow, (logo_x, logo_y), mask=glow)
            logo_img.paste(logo_with_border, (logo_x, logo_y),
                           mask=logo_with_border)

            # Combine the original image with the glow and the logo
            image_with_glow = Image.alpha_composite(
                image.convert("RGBA"), glow_img)
            final_image = Image.alpha_composite(image_with_glow, logo_img)

            # Convert back to RGB mode if the original image was in RGB
            return final_image.convert("RGB")

        except Exception as e:
            print(f"An error occurred while adding the logo: {e}")
            return image
