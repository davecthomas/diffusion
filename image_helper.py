import csv
import glob
import os
import re
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from datetime import datetime


class ImageHelper:
    def __init__(self, image_folder="images", logo_folder="logo"):
        # Timestamped log file
        csv_file = (f"image_log_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.image_folder = image_folder
        self.logo_folder = logo_folder
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.logo_folder, exist_ok=True)
        self.csv_file = os.path.join(
            self.image_folder, csv_file)

    def sanitize_filename(self, text: str) -> str:
        """
        Sanitizes a string to make it a valid filename by removing or replacing invalid characters.
        """
        return re.sub(r'[^\w\-_\.]', '_', text).strip()[:96]

    def save_raw_image(self, img_data: bytes, prompt: str, idx: int) -> str:
        """
        Saves the raw image data to a file with a date and timestamp prepended to the filename.

        Args:
            img_data (bytes): The raw image data to save.
            prompt (str): The prompt used to generate the image.
            idx (int): The index of the image (in case of multiple images for the same prompt).

        Returns:
            str: The filename where the image was saved.
        """
        # Sanitize the prompt to make it safe for a filename
        sanitized_prompt = self.sanitize_filename(prompt)

        # Get current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Prepend the timestamp to the filename
        raw_image_filename = f"{
            self.image_folder}/{timestamp}_{sanitized_prompt}_{idx + 1}_raw.png"

        # Save the raw image data
        with open(raw_image_filename, 'wb') as raw_img_file:
            raw_img_file.write(img_data)

        print(f"Raw image saved as {raw_image_filename}")
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
        return final_filename

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

    def evaluate_background_for_logo_selection(self, image: Image.Image) -> str:
        """
        Helper function that loads the logo, retrieves its dimensions (width and height), 
        and calls evaluate_background_brightness to determine the brightness of the region 
        where the logo will be placed.

        Args:
            image (Image.Image): The image on which the logo will be placed.

        Returns:
            str: A string indicating the brightness level of the background ("dark", "medium", or "light").
        """
        try:
            # Find the first logo file in the logo folder
            logo_files = glob.glob(os.path.join(self.logo_folder, "*"))
            if not logo_files:
                print("No logo files found.")
                return "no_logo"

            logo_path = logo_files[0]
            logo = Image.open(logo_path)

            # Get the width and height of the logo
            logo_width, logo_height = logo.size

            # Call the evaluate_background_brightness function to determine the background brightness
            return self.evaluate_background_brightness(image, logo_width, logo_height)

        except Exception as e:
            print(
                f"An error occurred while evaluating the logo background: {e}")
            return "error"

    def get_perceptual_brightness(self, rgb_pixel: tuple) -> float:
        """
        Calculates the perceptual brightness of an RGB pixel using weighted values for R, G, and B.

        Args:
            rgb_pixel (tuple): A tuple containing the (R, G, B) values of the pixel.

        Returns:
            float: The perceptual brightness value for the pixel.
        """
        r, g, b = rgb_pixel
        return 0.299 * r + 0.587 * g + 0.114 * b

    def evaluate_background_brightness(self, image: Image.Image, logo_width: int, logo_height: int) -> str:
        """
        Evaluates the brightness of the background region where the logo is placed (top center of the image).
        Returns "dark", "medium", or "light" based on the average perceptual brightness of that region.

        Args:
            image (Image.Image): The image on which the logo will be placed.
            logo_width (int): The width of the logo.
            logo_height (int): The height of the logo.

        Returns:
            str: A string indicating the brightness level of the background ("dark", "medium", or "light").
        """
        # Convert the image to RGB mode
        rgb_image = image.convert("RGB")

        # Define the region where the logo will be placed (top center)
        logo_x = (image.width - logo_width) // 2
        logo_y = 20  # Assuming 20 pixels margin from the top
        logo_region = rgb_image.crop(
            (logo_x, logo_y, logo_x + logo_width, logo_y + logo_height))

        # Calculate the average perceptual brightness of the region
        pixels = list(logo_region.getdata())
        total_brightness = sum(self.get_perceptual_brightness(pixel)
                               for pixel in pixels)
        avg_brightness = total_brightness / len(pixels)
        print(f"Average brightness of the logo region: {avg_brightness}")

        # Adjust brightness thresholds based on perceptual brightness
        if avg_brightness < 35:
            return "dark"
        elif 35 <= avg_brightness < 45:
            return "medium"
        else:
            return "light"

    def add_logo_to_image(self, image: Image.Image, image_lightness: str = "light") -> Image.Image:
        """
        Loads the first logo file from the logo folder, places it at the top center of the image, 
        and resizes the logo to fit comfortably (50% of the image width).

        Args:
            image (Image.Image): The image on which the logo will be placed.

        Returns:
            Image.Image: The modified image with the logo added.
        """
        try:
            if image_lightness == "dark":
                logo_path = os.path.join(self.logo_folder, "light_logo.png")
            else:
                logo_path = os.path.join(self.logo_folder, "dark_logo.png")
            # # Find the first logo file in the logo folder
            # logo_files = glob.glob(os.path.join(self.logo_folder, "*"))
            # if not logo_files:
            #     print("No logo files found. Returning the original image.")
            #     return image

            # logo_path = logo_files[0]
            logo = Image.open(logo_path)

            # Resize the logo to 50% of the image width
            max_logo_width = image.width // 2
            logo_ratio = logo.width / logo.height
            logo_height = int(max_logo_width / logo_ratio)
            logo = logo.resize((max_logo_width, logo_height),
                               Image.Resampling.LANCZOS)

            # Calculate the position to center the logo at the top of the image
            logo_x = (image.width - logo.width) // 2
            logo_y = 20  # Add some space from the top

            # Ensure the image is in RGBA mode for transparency
            image = image.convert("RGBA")

            # Paste the logo onto the image
            image.paste(logo, (logo_x, logo_y), mask=logo)

            # Convert the image back to RGB if needed
            return image.convert("RGB")

        except Exception as e:
            print(f"An error occurred while adding the logo: {e}")
            return image
