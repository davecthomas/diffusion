import csv
import glob
import io
import os
import re
from PIL import Image, ImageOps, ImageFilter, ImageDraw
# from IPython.display import Image as Img2, HTML, display
# import io
from datetime import datetime


class ImageHelper:
    def __init__(self, image_folder="images", logo_folder="logo", hex_mode=False):
        # Timestamped log file
        csv_file = (f"image_log_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.hex_mode = hex_mode
        if not hex_mode:
            self.image_folder = image_folder
            self.logo_folder = logo_folder
            os.makedirs(self.image_folder, exist_ok=True)
            os.makedirs(self.logo_folder, exist_ok=True)
            self.csv_file = os.path.join(
                self.image_folder, csv_file)
        else:
            self.csv_file = csv_file

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
        if not self.hex_mode:
            raw_image_filename = (
                f"{self.image_folder}/{timestamp}_{sanitized_prompt}_{idx + 1}_raw.png")
        else:
            raw_image_filename = (
                f"{timestamp}_{sanitized_prompt}_{idx + 1}_raw.png")

        # Save the raw image data
        with open(raw_image_filename, 'wb') as raw_img_file:
            raw_img_file.write(img_data)

        # print(f"Raw image saved as {raw_image_filename}")
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

    def save_image(self, img_data: bytes, filename: str):
        """
        Saves the image to the images/ directory with a date and timestamp prepended to the filename.
        """
        # Get current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Prepend the timestamp to the filename
        filename_with_timestamp = f"{timestamp}_{filename}"

        # Construct the full path for the file
        if not self.hex_mode:
            final_filename = os.path.join(
                self.image_folder, filename_with_timestamp)
        else:
            final_filename = filename_with_timestamp

        with open(final_filename, 'wb') as img_file:
            img_file.write(img_data)
        # print(f"Image saved as {final_filename}")
        return final_filename

    def log_to_csv(self, prompt: str, dimensions: tuple, filename: str, image_gen_time: float = 0, image_manip_time: float = 0):
        """
        Logs the prompt, dimensions, and filename to a CSV file with a header row.
        The header row is only written if the file is new.
        """
        file_exists = os.path.exists(self.csv_file)

        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write the header only if the file is new
            if not file_exists:
                writer.writerow(
                    ["Prompt", "Width", "Height", "Filename", "Image gen time", "Image manip time"])

            # Write the actual data
            writer.writerow([prompt, dimensions[0], dimensions[1],
                            filename, image_gen_time, image_manip_time])

        # print(f"Logged prompt and details to {self.csv_file}")

    def cleanup_csv_files(self):
        """
        Deletes all CSV files
        """
        if not self.hex_mode:
            png_file_pattern = os.path.join(self.image_folder, "*.png")
        else:
            png_file_pattern = "*.csv"

        files_to_delete = glob.glob(png_file_pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                # print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    def cleanup_png_files(self):
        """
        Deletes all PNG files except those that contain 'logo' in their filename.
        """
        if not self.hex_mode:
            png_file_pattern = os.path.join(self.image_folder, "*.png")
        else:
            png_file_pattern = "*.png"

        files_to_delete = glob.glob(png_file_pattern)

        for file_path in files_to_delete:
            # Skip files that end with '_logo.png'
            if os.path.basename(file_path).endswith('_logo.png'):
                continue
            try:
                os.remove(file_path)
                # print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    def cleanup_raw_files(self):
        """
        Deletes all files that contain 'raw.png' in their filename.
        """
        if not self.hex_mode:
            raw_file_pattern = os.path.join(self.image_folder, "*raw.png")
        else:
            raw_file_pattern = "*raw.png"
        files_to_delete = glob.glob(raw_file_pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                # print(f"Deleted {file_path}")
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
            logo_path = "medium_logo.png"
            logo = Image.open(logo_path)

            # Get the width and height of the logo
            logo_width, logo_height = logo.size

            # Call the evaluate_background_brightness function to determine the background brightness
            return self.evaluate_background_brightness(image, logo_width, logo_height)

        except Exception as e:
            # print(
            #     f"An error occurred while evaluating the logo background: {e}")
            return "medium"

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
        # print(f"Average brightness of the logo region: {avg_brightness}")

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
                if not self.hex_mode:
                    logo_path = os.path.join(
                        self.logo_folder, "light_logo.png")
                else:
                    logo_path = "light_logo.png"
            else:
                if not self.hex_mode:
                    logo_path = os.path.join(self.logo_folder, "dark_logo.png")
                else:
                    logo_path = "dark_logo.png"

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

    def create_download_button(file_name, button_text=None, hex_mode=False):
        """
        Generates an HTML button for downloading a file located in the current directory.

        :param file_name: Name of the file to download.
        :param button_text: Optional text to display on the button.
        :return: Displays an HTML button that triggers the download.
        """
        if button_text is None:
            button_text = f"Download {file_name}"
        html = f'''
        <a href="{file_name}" download="{file_name}">
            <button style="
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
            ">{button_text}</button>
        </a>
        '''
        if hex_mode:
            display(HTML(html))

    def display_image_in_hex(self, output_image, hex_mode=False, scale_percent=100):
        if not hex_mode:
            return
        """
        Displays a resized image in the Hex environment with optional scaling.

        :param output_image: Filename of the image to display.
        :param hex_mode: If True, displays the image within Hex.
        :param scale_percent: Percentage to scale the image size (e.g., 50 for 50%).
        """
        if not hex_mode:
            print("Hex mode is disabled. Image will not be displayed.")
            return

        # Check if the file exists in the current directory
        if not os.path.isfile(output_image):
            print(f"Error: The file '{
                  output_image}' does not exist in the current directory.")
            return

        # Validate scale_percent
        if not isinstance(scale_percent, (int, float)):
            print("Error: scale_percent must be a number representing the percentage.")
            return

        if not (0 < scale_percent <= 100):
            print("Error: scale_percent must be between 1 and 100.")
            return

        try:
            # Open the image to get its original dimensions
            with Image.open(output_image) as img:
                original_width, original_height = img.size
                # print(f"Original Size: {original_width}px (width) x {original_height}px (height)")

                # Calculate new dimensions based on scale_percent
                new_width = max(1, int(original_width * (scale_percent / 100)))
                new_height = max(
                    1, int(original_height * (scale_percent / 100)))
                # print(f"Scaled Size: {new_width}px (width) x {new_height}px (height)")

                # Resize the image using ANTIALIAS filter for high-quality downsampling
                img_resized = img.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS)

                # Save the resized image to a bytes buffer
                buffer = io.BytesIO()
                img_resized.save(buffer, format=img.format)
                buffer.seek(0)

                # Create an IPython.display.Image object from the buffer
                display_img = Img2(data=buffer.read())
                display(display_img)

        except Exception as e:
            print(f"An error occurred while processing the image: {e}")

    def get_image_file_paths(self, folder_path: str) -> list:
        """
        Returns a list of image file paths found in the given folder.
        """
        import os
        import glob

        # Define allowed image extensions
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')

        image_paths = []
        for extension in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, extension)))

        # Sort the list of image paths for consistency
        image_paths.sort()

        return image_paths

    def convert_to_png(self, image_filepath: str, max_size: int = 4 * 1024 * 1024, min_resize_factor: float = 0.4) -> str:
        """
        Converts a JPEG image to PNG format and ensures the resulting PNG file size does not exceed max_size.
        Utilizes color quantization first and, if necessary, incremental resizing to reduce file size efficiently 
        without significant loss of detail. If the image is already a PNG, it checks and optimizes the file size.

        Args:
            image_filepath (str): The file path to the image to be converted.
            max_size (int): The maximum allowed file size in bytes. Defaults to 4MB.

        Returns:
            str: The file path to the converted and optimized PNG image.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image format is not JPG, JPEG, or PNG.
            Exception: If an error occurs during the conversion or optimization process.
        """

        MB = 1024 * 1024
        if not os.path.isfile(image_filepath):
            raise FileNotFoundError(
                f"The file '{image_filepath}' does not exist.")

        file_extension = os.path.splitext(image_filepath)[1].lower()

        # Convert JPG/JPEG to PNG if necessary
        if file_extension in ['.jpg', '.jpeg']:
            try:
                with Image.open(image_filepath) as img:
                    new_filepath = os.path.splitext(image_filepath)[0] + '.png'
                    # Ensure RGBA mode for PNG quality
                    img = img.convert('RGBA')
                    img.save(new_filepath, 'PNG', optimize=True)
                    # print((f"Converted '{image_filepath}' to '"
                    #        f"{new_filepath}'."))
                    image_filepath = new_filepath  # Update filepath to the new PNG
            except Exception as e:
                print((f"An error occurred while converting '"
                       f"{image_filepath}' to PNG: {e}"))
                raise
        elif file_extension != '.png':
            raise ValueError(
                "Unsupported file format. Only JPG, JPEG, and PNG formats are supported.")

        # Step 1: Apply quantization
        try:
            current_size: int = os.path.getsize(image_filepath)
            if current_size > max_size:
                with Image.open(image_filepath) as img:
                    # Convert to RGB if necessary for MEDIANCUT quantization
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    quantized_img = img.quantize(
                        method=Image.MEDIANCUT, colors=256)
                    quantized_img.save(image_filepath, 'PNG', optimize=True)

            # Check size again after quantization
            original_size: int = current_size
            current_size = os.path.getsize(image_filepath)
            # print((f"Applied quantization; file size was {original_size / MB:.2f}MB. and is now "
            #        f"{current_size / MB:.2f}MB."))

            # Step 2: Incremental resizing if still above size limit
            if current_size > max_size:
                with Image.open(image_filepath) as img:
                    resize_factor = 0.95  # Start by reducing size by 5%

                    while current_size > max_size and resize_factor >= min_resize_factor:
                        # Calculate new dimensions
                        new_width = int(img.width * resize_factor)
                        new_height = int(img.height * resize_factor)
                        resized_img = img.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS)

                        resized_img.save(
                            image_filepath, 'PNG', optimize=True)

                        # Update file size
                        current_size = os.path.getsize(image_filepath)
                        # print((f"Resized and quantized image to {new_width}x"
                        #        f"{new_height} pixels; file size is now "
                        #        f"{current_size / MB:.2f}MB."))

                        # Reduce resize factor for next iteration if necessary
                        resize_factor -= 0.05

                    if current_size > max_size:
                        raise Exception((f"Unable to reduce the image size below "
                                         f"{max_size / MB:.2f}MB after resizing and optimization."))

                    # print((f"Final image saved as '{image_filepath}'. file size was {original_size / MB:.2f}MB. and is now "
                    #        f"{current_size / MB:.2f}MB."))

            return image_filepath

        except Exception as e:
            print((f"An error occurred while optimizing '"
                   f"{image_filepath}': {e}"))
            raise
