from groq import Groq
from diffusers import StableDiffusionPipeline
import torch

# Story generation using Groq API
api_key_groq = "your_groq_API_key"  # Replace with your Groq API Key
client = Groq(api_key=api_key_groq)

def generate_story(context):
    """
    Function to generate a story based on the provided context using Groq API.
    """
    try:
        # Call the Groq API to generate a story
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"Create a story based on this context: {context}"}
            ],
            model="llama3-8b-8192",
        )
        story = response.choices[0].message.content
        return story
    except Exception as e:
        return f"Error: {e}"

# Image generation using Stable Diffusion
def generate_image(prompt, output_path="output.png"):
    """
    Function to generate an image based on the prompt using Stable Diffusion locally.
    """
    try:
        model_id = "CompVis/stable-diffusion-v1-4"  # Hugging Face model ID
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Use GPU if available

        # Generate an image
        print("Generating image... (this may take a while)")
        image = pipe(prompt).images[0]
        image.save(output_path)
        return output_path
    except Exception as e:
        return f"Error: {e}"
