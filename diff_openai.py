import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from image_helper import ImageHelper


class DiffOpenAI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.image_model = os.getenv("OPENAI_DIFFUSION_MODEL", "dall-e-3")
        self.image_size_options = [(1024, 1024), (1024, 1792), (1792, 1024)]
        self.image_helper = ImageHelper()  # Create an instance of ImageHelper

    def get_closest_supported_size(self, width: int, height: int) -> tuple:
        """
        Finds the closest supported size based on the provided dimensions.
        """
        return min(self.image_size_options, key=lambda size: abs(size[0] - width) + abs(size[1] - height))

    def generate_image(self, prompt: str, dimensions: tuple = (1024, 1024), num_images=1) -> bytes:
        """
        Generates an image based on the given prompt using OpenAI's diffusion model, returning the raw image data.
        """
        closest_size = self.get_closest_supported_size(*dimensions)

        try:
            # Make the API request to generate an image
            response = self.client.images.generate(
                model=self.image_model,
                prompt=prompt,
                n=num_images,
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
        system_prompt = ("You are a prompt creator for a visual artist. "
                         "You avoid unsafe prompts that violate OpenAI's policy. "
                         "Your prompts should be varied, inspiring, and positive, covering a broad artistic range "
                         "including, but not limited to photography, painting, printmaking, and nostalgic lithographs. "
                         "Do not name any living or deceased artists or other persons in your prompts. "
                         "You should only return the prompt itself and no extraneous text.")
        for _ in range(num_prompts):
            # Use sendPrompt to generate a new prompt
            generated_prompt = self.send_prompt(system_prompt,
                                                (f"Generate a unique prompt that exclusively focuses on this subject: "
                                                 f"'{seed_prompt}'."
                                                 "Describe interesting and creative camera angles, the angle and color of natural and artificial lighting, "
                                                 "times of day, weather, artistic styles, and eras. "
                                                 "Choose a color palette that is consistently carried throughout the image. "
                                                 "Try to capture the detail of the design and textures of the walls, floor coverings, and ceilings. "
                                                 "Work to capture the beauty, the vitality, history, and nostalgia of the subject. "
                                                 "The prompt should explicitly state that the scene should not have any text or words in it."))
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
