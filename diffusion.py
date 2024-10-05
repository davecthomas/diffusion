from diff_openai import DiffOpenAI


if __name__ == "__main__":
    num_prompts = 5
    img_width, img_height = 1024, 1792
    diff_openai = DiffOpenAI()
    seed_prompt = ("a gas station with no people or cars; not necessarily abandoned, "
                   "but lacking current activity")
    prompts = diff_openai.generate_image_prompts(seed_prompt, num_prompts)

    # Loop through each prompt, generate the image, crop it, and save it
    for prompt in prompts:
        try:
            print(f"Generating image for prompt: {prompt}")
            img_data = diff_openai.generate_image(
                prompt, dimensions=(img_width, img_height))

            # Save raw image using the new method
            raw_image_filename = diff_openai.save_raw_image(
                img_data, prompt, idx=0)  # Using 0 as the index for simplicity

            # Crop the raw image (No need for Image.open in main)
            cropped_img = diff_openai.crop_image(
                raw_image_filename, img_width, img_height)
            final_image_filename = f"{
                diff_openai.sanitize_filename(prompt)}_1024x1792.png"
            diff_openai.save_image(cropped_img, final_image_filename)

            # Log the details to the CSV file
            diff_openai.log_to_csv(
                prompt, (img_width, img_height), final_image_filename)

        except Exception as e:
            print(f"An error occurred: {e}")

    # Cleanup raw image files
    diff_openai.cleanup_raw_files()
