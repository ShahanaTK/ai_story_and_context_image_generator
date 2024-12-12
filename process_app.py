import streamlit as st
from process import generate_story, generate_image

# Streamlit app layout
st.title("Story Generator with Image ✨📜🎨")
st.write("Create a unique story and get an image generated based on the story context!")

# User input for story context
story_context = st.text_area(
    "Enter the context or idea for your story:", 
    placeholder="Example: A brave knight fighting a fire-breathing dragon to save a kingdom..."
)

# Generate button
if st.button("Generate Story and Image"):
    if not story_context.strip():
        st.warning("Please enter a context for the story.")
    else:
        # Generate the story
        with st.spinner("Generating your story..."):
            story = generate_story(story_context)
        
        # Display the story
        if "Error" in story:
            st.error(f"Story Generation Failed: {story}")
        else:
            st.subheader("Generated Story:")
            st.write(story)
            
            # Generate the image based on the story
            with st.spinner("Generating an image based on the story..."):
                image_path = generate_image(story_context)  # Use story context as image prompt

            # Display the image
            if "Error" in image_path:
                st.error(f"Image Generation Failed: {image_path}")
            else:
                st.subheader("Generated Image:")
                st.image(image_path, caption="Image Generated by Stable Diffusion")
