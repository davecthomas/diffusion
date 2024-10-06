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

    def evaluate_background_brightness(self, image: Image.Image, logo_width: int, logo_height: int) -> str:
        """
        Evaluates the brightness of the background region where the logo is placed (top center of the image).
        Returns "dark", "medium", or "light" based on the average brightness of that region.

        Args:
            image (Image.Image): The image on which the logo will be placed.
            logo_width (int): The width of the logo.
            logo_height (int): The height of the logo.

        Returns:
            str: A string indicating the brightness level of the background ("dark", "medium", or "light").
        """
        # Convert the image to grayscale to evaluate brightness
        grayscale_image = image.convert("L")

        # Define the region where the logo will be placed (top center)
        logo_x = (image.width - logo_width) // 2
        logo_y = 20  # Assuming 20 pixels margin from the top
        logo_region = grayscale_image.crop(
            (logo_x, logo_y, logo_x + logo_width, logo_y + logo_height))

        # Calculate the average brightness of the region
        pixels = list(logo_region.getdata())
        avg_brightness = sum(pixels) / len(pixels)

        # Define brightness thresholds
        if avg_brightness < 85:
            return "dark"
        elif 85 <= avg_brightness < 170:
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

    def add_glow_logo_to_image(self, image: Image.Image) -> Image.Image:
        """
        Adds a white glow behind where the logo would be placed and ensures that the glow diffuses naturally.

        Args:
            image (Image.Image): The image on which the glow and logo will be placed.

        Returns:
            Image.Image: The modified image with the glow and logo added.
        """
        try:
            # Find the first logo file in the logo folder
            logo_files = glob.glob(os.path.join(self.logo_folder, "*"))
            if not logo_files:
                print("No logo files found. Returning the original image.")
                return image  # Return the original image unchanged if no logo is found

            logo_path = logo_files[0]
            logo = Image.open(logo_path)

            # Resize the logo to fit comfortably at the top of the image (50% of image width)
            max_logo_width = image.width // 2
            logo_ratio = logo.width / logo.height
            logo_height = int(max_logo_width / logo_ratio)
            logo = logo.resize((max_logo_width, logo_height),
                               Image.Resampling.LANCZOS)

            # Ensure the image is in RGBA mode
            image = image.convert("RGBA")

            # Create a larger canvas for the glow to diffuse naturally
            # Add more space for the glow
            glow_canvas_size = (logo.width + 600, logo.height + 400)
            # Transparent background for glow
            glow = Image.new("RGBA", glow_canvas_size, (0, 0, 0, 0))

            # Draw a white shape (ellipse) in place of the logo to create the glow
            draw = ImageDraw.Draw(glow)
            draw.ellipse([(100, 100), (glow_canvas_size[0]-100,
                                       glow_canvas_size[1]-100)], fill=(255, 255, 255, 255))

            # Apply the Gaussian blur for the glow effect
            glow = glow.filter(ImageFilter.GaussianBlur(100))  # Stronger blur

            # Create a new RGBA image for the glow layer
            glow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))

            # Center the glow at the top of the image
            glow_x = (image.width - glow_canvas_size[0]) // 2
            glow_y = 20  # Add space from the top

            # Paste the glow onto the glow layer
            glow_layer.paste(glow, (glow_x, glow_y), mask=glow)

            # Combine the original image with the glow layer
            image_with_glow = Image.alpha_composite(image, glow_layer)

            # LOGO CODE COMMENTED OUT BELOW:
            logo_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            # Adjust logo position based on glow
            logo_layer.paste(logo, (glow_x + 100, glow_y + 100), mask=logo)
            final_image = Image.alpha_composite(image_with_glow, logo_layer)
            return final_image.convert("RGB")

            # Return the image with the glow only (no logo)
            # return image_with_glow.convert("RGB")

        except Exception as e:
            print(f"An error occurred while applying the glow: {e}")
            return image

    def test_glow_generation(self, image: Image.Image) -> Image.Image:
        """
        Test function to create a glow around a simple white shape (circle) to verify if the glow effect works.
        This method saves the glow separately to verify if it's being generated correctly.

        Args:
            image (Image.Image): The image on which the glow would be placed.

        Returns:
            Image.Image: The original image (for testing purposes).
        """
        try:
            # Create a simple white circle as the 'logo' for testing
            logo_size = (800, 200)  # Test with a simple white circle
            logo = Image.new("RGBA", logo_size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(logo)
            draw.ellipse((0, 0, logo_size[0], logo_size[1]), fill=(
                255, 255, 255, 255))

            # Create a larger canvas for the glow to diffuse naturally
            # Add space for the glow
            glow_canvas_size = (logo.width + 200, logo.height + 200)
            glow = Image.new("RGBA", glow_canvas_size, (0, 0, 0, 0))

            # Position the circle in the middle of the canvas for the glow
            logo_x_on_glow = (glow_canvas_size[0] - logo.width) // 2
            logo_y_on_glow = (glow_canvas_size[1] - logo.height) // 2
            glow.paste(logo, (logo_x_on_glow, logo_y_on_glow), mask=logo)

            # Apply a strong Gaussian blur to create the glow
            glow = glow.filter(ImageFilter.GaussianBlur(50))

            # Save the glow layer to inspect it
            glow.save("test_glow.png")

            print("Glow saved as 'test_glow.png'. Please inspect the output.")
            return image  # Return original image for now

        except Exception as e:
            print(f"An error occurred while generating the glow: {e}")
            return image
