import base64
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import requests
from tqdm import tqdm
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
        Generates an image based on the given prompt using OpenAI's diffusion model, returning 
        a list of dictionaries containing the prompt, revised prompt, and image data.
        """
        closest_size = self.get_closest_supported_size(*dimensions)

        try:
            # The API only supports generating one image at a time, so we must loop to generate multiple images
            img_data_list_dict = []
            for _ in tqdm(range(num_images), desc="Generating Images", unit="image"):
                response = self.client.images.generate(
                    model=self.image_model,
                    prompt=prompt,
                    n=1,
                    size=f"{closest_size[0]}x{closest_size[1]}"
                )

                image_data = response.data[0]
                img_data = requests.get(image_data.url).content
                dict_image_data = {
                    "prompt": prompt,
                    "revised_prompt": image_data.revised_prompt,
                    "image_data": img_data,
                }
                img_data_list_dict.append(dict_image_data)

            return img_data_list_dict

        except Exception as e:
            print(f"An error occurred while generating the image: {e}")
            raise

    def generate_image_prompts(self, seed_prompt: str, num_prompts: int) -> list:
        prompts = []
        system_prompt = ("You are a prompt creator for a visual artist. "
                         "You avoid unsafe prompts that violate OpenAI's policy. "
                         "Your prompts should be varied, inspiring, and positive, covering a broad artistic range "
                         "including, but not limited to photography, painting, and various printmaking techniques. "
                         "Do not name any living or deceased artists or other persons in your prompts. "
                         "You should only return the prompt itself and no extraneous text.")
        for _ in range(num_prompts):
            # Use sendPrompt to generate a new prompt
            generated_prompt = self.send_prompt(system_prompt,
                                                (f"Generate a unique prompt that exclusively focuses on this subject: "
                                                 f"'{seed_prompt}'."
                                                 "Describe interesting and creative camera angles, the angle and color of natural and artificial lighting, "
                                                 "times of day, weather, artistic styles, and eras, including midcentury through modern times. "
                                                 "Choose a color palette that is consistently carried throughout the image. "
                                                 "Try to capture the detail of the design and textures of the walls, floor coverings, and ceilings. "
                                                 "Work to capture the beauty, the vitality, and history of the subject. "
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

    def get_image_style_description(self, image_path: str) -> str:
        """
        Uses the OpenAI API to get a detailed style description of the image.
        """
        try:
            # Encode the image in base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(
                    image_file.read()).decode('utf-8')

            # Create the message with the image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the visual style and characteristics of this image in detail, including lighting, color palette, perspective, camera placement, subject, contrast, textures, and overall atmosphere. Focus on stylistic elements."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ]

            # Send the message to the OpenAI API using self.client
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            # Extract and return the assistant's reply
            if response.choices and response.choices[0].message.content is not None:
                return response.choices[0].message.content.strip()
            else:
                return ""

        except Exception as e:
            print((f"Error generating style description for image "
                  f"{image_path}: {e}"))
            return ""

    def get_combined_style_description(self, image_paths: list) -> str:
        """
        Uses the OpenAI API to get a combined style description from multiple images.
        """
        try:
            # Prepare the content with the initial prompt
            content = [
                {
                    "type": "text",
                    "text":   "Generate a detailed style description of the common attributes of style that all the following images convey, "
                    "including specific independent sections describing the following: "
                    "1. For the primary human focus in the image, the typical posing, framing, proximity to the camera, clothing style, and importantly, if their head or face is visible. "
                    "2. Typical lighting, light color, lighting angle, and time of day, "
                    "3. Common color palette and dominant colors. Do not list the objects, just describe the colors. For example, do not say 'blue car' rather say 'dominant color is blue'."
                    "4. Common camera angles, image orientation, camera placement, and perspective"
                    "5. Mood, atmosphere, and artistic style, including any specific eras or artistic movements, if any,"
                    "6. Image processing, such as contrast, boosting or other effects, "
                    "7. Cleanliness of the surroundings, "
                    "9. Common textures prevalent, if any, in surfaces and clothing."
                    "Only include stylistic elements common to all images. Do not convey specifics of any single image."
                    "Avoid unsafe language that violate OpenAI's policy or may cause prompt truncation."
                }
            ]

            # Encode each image in base64 and add to content
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(
                        image_file.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "high"
                        }
                    })

            # Create the message with the content
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            # Send the message to the OpenAI API using self.client
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            if response.choices and response.choices[0].message.content is not None:
                return response.choices[0].message.content.strip()
            else:
                return ""

        except Exception as e:
            print(f"Error generating combined style description: {e}")
            return ""

    def merge_style_descriptions(self, style_descriptions: list) -> str:
        """
        Merges multiple style descriptions into a single detailed style prompt for image generation,
        removing any information about the subject and any duplicate information.
        """
        system_prompt = "You are an expert in analyzing and merging image style descriptions for AI image generation."
        user_prompt = (
            "Given the following style descriptions:\n\n"
            f"{' '.join(style_descriptions)}\n\n"
            "Merge these descriptions into a single, comprehensive style prompt suitable for guiding an AI image generator. "
            "The merged style should be detailed, cohesive, and include stylistic elements such as dominant color scheme, lighting, color palette, perspective, camera placement, contrast, textures, and overall atmosphere. "
            "For pictures including people, note the parts of their bodies that are not visible, their poses, and posture. "
            "With the exception of people, remove any references to specific subjects. Eliminate any duplicate style information. "
        )

        # Generate the merged style description using self.client
        merged_style_description = self.send_prompt(system_prompt, user_prompt)
        return merged_style_description.strip()

    def generate_image_with_style_references(
        self, image_filenames: list, prompt: str, dimensions: tuple = (1024, 1024), num_images=1
    ) -> bytes:
        """
        Generates an image based on the given prompt and a list of image filenames to reference their styles.
        """
        # Get style descriptions for each image
        style_description: str = ""
        style_description = self.get_combined_style_description(
            image_filenames)

        # Create a combined prompt
        combined_prompt = self.combine_prompt_with_styles(
            prompt, style_description)

        # Generate image using the combined prompt
        list_dict_image_data = self.generate_image(
            combined_prompt, dimensions=dimensions, num_images=num_images)

        return list_dict_image_data

    def combine_prompt_with_styles(self, content_prompt: str, style_descriptions: str) -> str:
        """
        Uses the OpenAI API to combine the content prompt with style descriptions into a detailed prompt.
        """
        system_prompt = "You are an expert prompt engineer for AI image generation."
        user_prompt = (
            f"Based on the following style descriptions: {
                style_descriptions}, "
            f"and the following content prompt: '{content_prompt}', "
            "generate a detailed image generation prompt that incorporates all the style elements, "
            "including specific independent sections describing the following: "
            "1. Primary human subject, if any, their pose, proximity to the camera, posture, clothing, and importantly, if their head or face is visible. "
            "2. Lighting, light color, lighting angle, and time of day, "
            "3. Color palette and dominant colors. Do not list the objects, just describe the colors. For example, do not say 'blue car' rather say 'dominant color is blue'."
            "4. Camera angle, image orientation, camera placement, and perspective"
            "5. Subject, mood, atmosphere, and artistic style, including any specific eras or artistic movements, if any,"
            "6. Image processing, such as contrast, boosting or other effects, "
            "7. Cleanliness of the subject and surroundings and any other relevant details. "
            "8. Object of focus. Describe what the primary subject is loking at, what they are holding, such as a phone, what is on screen, and what direction the phone is facing. "
            "9. Textures of surfaces. Do not list the objects, just their textures. For example, don't say 'wooden table' rather say 'wooden textures are prevalent'."
            "The resulting image generation prompt should be detailed and descriptive. No text is allowed in the image."
            "The resulting image generation prompt should exclude descriptions that don't relate to the provided content prompt."
        )

        # Generate the combined prompt using self.client
        combined_prompt = self.send_prompt(system_prompt, user_prompt)
        self.combined_prompt = combined_prompt.strip()
        return self.combined_prompt

    def get_image_variations(self, reference_image_filename: str, num_variations: int, dimensions: tuple = (1024, 1024),) -> list:
        """
        Generates variations of a given image using OpenAI's image variation API.

        Parameters:
            image_filename (str): The file path to the original image. Must be a valid PNG file, 
                                less than 4MB in size, and square.
            num_variations (int): The number of image variations to generate. Must be between 1 and 10.

        Returns:
            list: A list of dictionaries, each containing:
                - "filename": The filename for the variation.
                - "img_data": The byte stream of the generated image.
        """
        # Validate input parameters
        if not os.path.isfile(reference_image_filename):
            raise FileNotFoundError(
                f"The file {reference_image_filename} does not exist.")

        closest_size = self.get_closest_supported_size(*dimensions)

        # Ensure num_variations is within the allowed range
        if not (1 <= num_variations <= 10):
            raise ValueError("num_variations must be between 1 and 10.")

        list_dict_img_variations = []

        # Loop to generate each variation
        for i in tqdm(range(num_variations), desc="Generating Image Variations", unit="variation"):
            try:
                with open(reference_image_filename, 'rb') as reference_img_file:
                    response = self.client.images.create_variation(
                        image=reference_img_file,
                        n=1,
                        size=f"{closest_size[0]}x{closest_size[1]}",
                        response_format="url"
                    )

                image_data = response.data[0]
                img_data = requests.get(image_data.url).content

                # Generate a unique filename for the variation
                original_basename = os.path.basename(reference_image_filename)
                variation_filename = f"variation_{i+1}_{original_basename}"

                # Append the variation data to the list
                list_dict_img_variations.append({
                    "original_filename": original_basename,
                    "variation_filename": variation_filename,
                    "image_data": img_data
                })

            except Exception as e:
                print(
                    f"An error occurred while generating variation {i+1}: {e}")
                continue

        return list_dict_img_variations


def generate_images_from_style():
    hex_mode = False
    img_width, img_height = 1024, 1792

    diff_openai = DiffOpenAI()
    image_helper = ImageHelper(hex_mode=hex_mode)

    # List of image filenames to reference styles
    # Define the folder containing style reference images
    style_image_folder = "./style_ref_images"

    # Get list of image files from the specified folder
    style_image_filenames = image_helper.get_image_file_paths(
        style_image_folder)

    # Check if any style images were found
    if not style_image_filenames:
        print((f"No style reference images found in "
               f"{style_image_folder}. Exiting."))
        exit()

    # New content prompt
    prompt = "A shopper at a grocery store checkout line. The shopper uses their mobile phone to pay. They are buying milk and packaged cookies."
    print((f"Generating image with prompt:\n "
           f"\t'{prompt}'\nand based on image styles found in {style_image_folder}..."))

    # Generate an image with style references
    try:
        # Get style descriptions for each image
        style_description: str = ""
        style_description = diff_openai.get_combined_style_description(
            style_image_filenames)

        # Create a combined prompt
        combined_prompt = diff_openai.combine_prompt_with_styles(
            prompt, style_description)

        list_dict_image_data = diff_openai.generate_image(
            prompt=combined_prompt,
            dimensions=(img_width, img_height),
            num_images=4  # Generating 4 images
        )
        # Store the combined prompt to a date-stamped text file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prompt_filename = f"./images/combined_prompt_{timestamp}.txt"
        with open(prompt_filename, 'w') as f:
            f.write(f"Root prompt: {prompt}\n\n")
            f.write(f"Source images:\n")
            f.write(f"""{style_image_filenames}\n\n""")
            f.write(f"Style descriptions:\n")
            f.write(f"""{style_description}\n\n""")

            # Save and process each image
            for idx, dict_img_data in enumerate(tqdm(list_dict_image_data, desc="Saving Images", unit="image", leave=True)):
                raw_image_filename = image_helper.save_raw_image(
                    dict_img_data.get("image_data", ""), prompt, idx=idx
                )
                # final_image_filename = (f"logo_{image_helper.sanitize_filename(prompt)}_{idx}_"
                #                         f"{img_width}x{img_height}.png")
                # image_helper.save_image(dict_img_data.get(
                #     "image_data", ""), final_image_filename)
                print((f"{idx} image for prompt "
                       f"{prompt} saved as {raw_image_filename}"))
                f.write(f"Image {idx+1}:\n")
                f.write("Original image prompt:\n")
                f.write(dict_img_data.get("prompt", ""))
                f.write("\n\nAltered prompt:\n")
                f.write(dict_img_data.get("revised_prompt", ""))
                f.write("\n\n")
        print(f"Image prompts saved as {prompt_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")


def generate_images_from_reference_image():
    hex_mode = False
    img_width, img_height = 1024, 1024

    diff_openai = DiffOpenAI()
    image_helper = ImageHelper(hex_mode=hex_mode)

    # List of image filenames to reference styles
    # Define the folder containing style reference images
    style_image_folder = "./style_ref_images"

    # Get list of image files from the specified folder
    style_image_filenames = image_helper.get_image_file_paths(
        style_image_folder)

    # Check if any style images were found
    if not style_image_filenames:
        print((f"No style reference images found in "
              f"{style_image_folder}. Exiting."))
        exit()

    for image in tqdm(style_image_filenames, desc="Getting image variations", unit="image"):
        image = image_helper.convert_to_png(image)
        list_dict_img_variations = diff_openai.get_image_variations(
            image, num_variations=4, dimensions=(img_width, img_height))
        for idx, dict_img_variations in enumerate(tqdm(list_dict_img_variations, desc="Saving Images", unit="image", leave=True)):
            variation_filename = f"{dict_img_variations.get(
                "variation_filename", "variation")}"
            raw_image_filename = image_helper.save_raw_image(dict_img_variations.get(
                "image_data", ""), variation_filename, idx)
            print(f"Image {idx+1} saved as {raw_image_filename}")


if __name__ == "__main__":
    generate_images_from_style()
    # generate_images_from_reference_image()
