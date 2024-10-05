import csv
import glob
from io import BytesIO
import os
import re
import requests
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageOps
from datetime import datetime


class DiffOpenAI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.image_model = os.getenv("OPENAI_DIFFUSION_MODEL", "dall-e-3")
        self.image_size_options = [(1024, 1024), (1024, 1792), (1792, 1024)]
        # Timestamped log file
        self.csv_file = f"image_log_{
            datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.backoff_delays = [1, 2, 4, 8, 16]  # unused
        # Ensure the images folder exists
        self.image_folder = "images"
        os.makedirs(self.image_folder, exist_ok=True)

    def sanitize_filename(self, text: str) -> str:
        """
        Sanitizes a string to make it a valid filename by removing or replacing invalid characters.
        """
        return re.sub(r'[^\w\-_\.]', '_', text).strip()[:128]

    def get_closest_supported_size(self, width: int, height: int) -> tuple:
        """
        Finds the closest supported size based on the provided dimensions.
        """
        return min(self.image_size_options, key=lambda size: abs(size[0] - width) + abs(size[1] - height))

    def crop_image(self, image_path: str, target_width: int, target_height: int) -> Image.Image:
        """
        Opens the image from the specified path, crops it to the target dimensions, and returns the cropped image.
        """
        try:
            # Open the image
            with Image.open(image_path) as img:
                # Crop the image to the specified dimensions
                cropped_img = ImageOps.fit(
                    img, (target_width, target_height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            return cropped_img
        except Exception as e:
            print(f"Error cropping image {image_path}: {e}")
            raise

    def save_image(self, img: Image.Image, filename: str):
        """
        Saves the image to a file.
        """
        final_filename = os.path.join(self.image_folder, filename)
        img.save(final_filename)
        print(f"Image saved as {final_filename}")

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
            print(f"Raw image saved as {raw_image_filename}")
        return raw_image_filename

    def log_to_csv(self, prompt: str, dimensions: tuple, filename: str):
        """
        Logs the prompt, dimensions, and filename to a CSV file with a header row.
        """
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Always write the header since the CSV file has a timestamp and is unique
            writer.writerow(["Prompt", "Width", "Height", "Filename"])
            # Log the actual data
            writer.writerow([prompt, dimensions[0], dimensions[1], filename])
        print(f"Logged prompt and details to {self.csv_file}")

    def generate_image(self, prompt: str, dimensions: tuple = (1024, 1024)) -> bytes:
        """
        Generates an image based on the given prompt using OpenAI's diffusion model, returning the raw image data.
        """
        closest_size = self.get_closest_supported_size(*dimensions)

        try:
            # Make the API request to generate an image
            response = self.client.images.generate(
                model=self.image_model,
                prompt=prompt,
                n=1,
                size=f"{closest_size[0]}x{closest_size[1]}"
            )

            image_data = response.data[0]
            image_url = image_data.url
            img_data = requests.get(image_url).content
            return img_data

        except Exception as e:
            print(f"An error occurred while generating the image: {e}")
            raise

    def generate_image_prompts(self, seed_prompt: str, num_prompts: int) -> list:
        prompts = []
        system_prompt = ("You are a prompt creator for a visual artist. You avoid unsafe prompts that violate OpenAI's policy. "
                         "Your prompts should be varied, inspiring, and positive, covering a broad artistic range."
                         "Do not name any living or deceased artists or other persons in your prompts. "
                         "You should only return the prompt itself and no extraneous text.")
        for _ in range(num_prompts):
            # Use sendPrompt to generate a new prompt
            generated_prompt = self.send_prompt(system_prompt,
                                                (f"Generate a unique prompt that exclusively focuses on this subject: "
                                                 f"'{seed_prompt}'."
                                                 "Vary the described camera angle, angle of natural and artificial lighting, time of day, artistic style, and era."
                                                 "Work to capture the beauty, the history, and nostalgia of the subject."))
            prompts.append(generated_prompt)

        return prompts

    def send_prompt(self, system_prompt, prompt: str) -> str:
        """
        Sends a prompt to the OpenAI API for chat and returns the completion result.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while sending the prompt: {e}")
            raise
