import google.generativeai as genai
from pathlib import Path
import streamlit as st
import json
import time
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure GenAI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to initialize the model
def initialize_model():
    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    return genai.GenerativeModel("gemini-1.5-pro", generation_config=generation_config)

# (rest of your code remains unchanged)
# Function to clean the JSON response
def clean_json_response(response_text):
    # Remove 'json' and backticks if present
    cleaned_text = re.sub(r'^json\s*```|```$', '', response_text, flags=re.IGNORECASE | re.MULTILINE)
    # Remove any leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to process the image and extract hemodialysis data
def extract_hemodialysis_data(model, image_path):
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_path.read_bytes()
    }
    
    prompt = """
    Analyze this image of a hemodialysis machine display. Extract all visible numerical data and parameters.
    Return the data in a JSON format with keys being the parameter names and values being the numerical readings.
    If you're unsure about a value, use null. Ensure your response is valid JSON without any additional text or formatting. Example format:
    {
        "Temperature": 30,
        "Conductivity": 14,
        "Blood Flow Rate": null,
        "Dialysate Flow Rate": 500
    }
    """
    
    start_time = time.time()
    response = model.generate_content([prompt, image_part])
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if response.text:
        cleaned_response = clean_json_response(response.text)
        try:
            data = json.loads(cleaned_response)
            return {"success": True, "data": data}, execution_time
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse JSON response", "raw_response": cleaned_response}, execution_time
    else:
        return {"success": False, "error": "No valid content generated"}, execution_time

# Streamlit app
def main():
    st.title("Hemodialysis Data Extractor")

    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image of a hemodialysis machine display", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Initialize the model
        model = initialize_model()
        
        # Button to extract data
        if st.button("Extract Data"):
            # Extract data based on the uploaded image
            image_path = Path("temp_image.jpg")
            result, execution_time = extract_hemodialysis_data(model, image_path)
            
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            
            # Display the extracted data or error message
            st.write("Extraction Result:")
            if result["success"]:
                st.json(result["data"])
            else:
                st.error(f"Error: {result['error']}")
                if "raw_response" in result:
                    st.write("Raw response from the model:")
                    st.code(result["raw_response"], language="json")
            
            # Display execution time
            st.write(f"Execution Time: {execution_time:.2f} seconds")
        
        # Optionally remove the temporary file
        Path("temp_image.jpg").unlink()

if __name__ == "__main__":
    main()