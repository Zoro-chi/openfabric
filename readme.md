# AI Creative Studio

AI Creative Studio is an application that allows users to generate both images and 3D models from text descriptions using a creative pipeline powered by Openfabric's AI services.

## Hugging Face Spaces Deployment

This repository is configured to be deployed as a Hugging Face Space. The main files for the Space deployment are:

- `app.py` - The main application file for Hugging Face Spaces
- `requirements.txt` - The dependencies needed for the Space
- `.env` - Environment variables for the Space

## Features

- Text-to-Image generation using Openfabric AI
- Image-to-3D model conversion
- Interactive 3D model viewer
- Gallery for browsing generated content

## How to Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)

   - Select "Gradio" as the SDK
   - Choose a name for your Space

2. Upload your files to the Space:

   - `app.py`
   - `requirements.txt`
   - `.env` (with your Openfabric API keys)
   - Your entire `app` directory with core modules

3. The Space will automatically build and deploy your application

## Local Development

To run the application locally:

1. Set up your environment variables in `.env`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Project Structure

- `app/` - Core application code
  - `core/` - Pipeline components
  - `data/` - Storage for generated images and models
- `app.py` - Main application entry point for Spaces

## Credits

Built using Gradio and Openfabric AI.
