from diff_openai import DiffOpenAI
from image_helper import ImageHelper
import time


if __name__ == "__main__":
    num_prompts = 5
    img_width, img_height = 1024, 1792
    diff_openai = DiffOpenAI()
    image_helper = ImageHelper()

    # List of seed prompts related to different locations
    seed_prompts = [
        "close-in exterior view of a quiet and clean gas station with no people or cars; not abandoned, but lacking current customer activity",
        "interior view of a neighborhood grocery store produce section showcasing beautiful and healthy produce, close-up. No shopping carts nor shoppers are visible",
        "interior view of a brightly lit and clean corner convenience store with no shoppers or employees visible",
        "interior view of a cozy coffee shop scene with no customers.",
        "interior view of a quaint, small, not fancy, but clean restaurant with no customers or people in the scene",
    ]

    # Iterate through each seed prompt
    for seed_prompt in seed_prompts:
        print(f"\nProcessing seed prompt: {seed_prompt}\n")

        # Generate image prompts for each seed prompt
        prompts = diff_openai.generate_image_prompts(seed_prompt, num_prompts)

        # Loop through each generated prompt, generate the image, crop it, and save it
        for i, prompt in enumerate(prompts):
            try:
                print(f"\n\tGenerating image for prompt:\n\t{prompt}\n")
                # Start timer for the total time of this iteration
                iteration_start_time = time.time()

                # Start timer for the image generation
                image_gen_start_time = time.time()

                print(f"\n\tGenerating image for prompt:\n\t{prompt}\n")
                img_data = diff_openai.generate_image(
                    prompt, dimensions=(img_width, img_height))

                # End timer for image generation
                image_gen_end_time = time.time()
                image_gen_duration = image_gen_end_time - image_gen_start_time
                print(f"Image generation time: {
                      image_gen_duration:.2f} seconds")

                # Save raw image using ImageHelper
                raw_image_filename = image_helper.save_raw_image(
                    img_data, seed_prompt, idx=i)

                # Crop the raw image using ImageHelper
                cropped_img = image_helper.crop_image(
                    raw_image_filename, img_width, img_height)
                image_lightness = image_helper.evaluate_background_for_logo_selection(
                    cropped_img)
                print(f"Image lightness: {image_lightness}")
                # Add the logo to the cropped image
                image_with_logo = image_helper.add_logo_to_image(
                    cropped_img, image_lightness)
                image_manipulation_end_time = time.time()
                image_manipulation_duration = image_manipulation_end_time - image_gen_end_time
                print(
                    f"Image manipulation time: {image_manipulation_duration:.2f} seconds")
                final_image_filename = (f"logo_{image_helper.sanitize_filename(seed_prompt)}_{i}_"
                                        f"{img_width}x{img_height}.png")
                final_image_filename = image_helper.save_image(
                    image_with_logo, final_image_filename)

                # Log the details to the CSV file using ImageHelper
                image_helper.log_to_csv(
                    prompt, (img_width, img_height), final_image_filename)

            except Exception as e:
                print(f"An error occurred: {e}")

        # Cleanup raw image files after processing each seed prompt
        image_helper.cleanup_raw_files()
