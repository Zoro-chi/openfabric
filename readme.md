# AI Creative

AI Creative is an end-to-end pipeline for generating 3D models from text prompts, leveraging cutting-edge AI technologies including text-to-image generation and image-to-3D model transformation.

<img width="1470" alt="AI Creative Studio" src="https://github.com/user-attachments/assets/7db5ac5c-f707-433b-864f-5b165021de9a" />

## Features

- **Text-to-Image Generation**: Convert descriptive text prompts into detailed, high-quality images
- **Text Prompt Enhancement**: Automatically enhance user prompts for better results using LLM
- **Image-to-3D Conversion**: Transform 2D images into textured 3D models in GLB format
- **Video Preview**: Generate video previews of the 3D models for quick visualization
- **User-friendly Interface**: Simple web UI for interacting with the application
- **Openfabric Integration**: Leverages Openfabric's powerful AI applications ecosystem

## System Requirements

- Python 3.10+
- 8GB RAM (16GB+ recommended for larger models)
- Modern CPU (GPU recommended for improved performance)
- 1GB free disk space for the application and generated assets
- Internet connection (for Openfabric API access)
- Operating Systems: Windows 10/11, macOS, Linux

## Default Model Used

- The LLM used is the `meta-llama/Llama-3.2-3B-Instruct` model, which is a lightweight version of the Llama-3.2 model. It is designed to run on consumer-grade hardware with 8GB of RAM. The model is capable of enhancing text prompts for better image generation results.

- This is a gated model; you would need your huggingface token to access it. You can request access to the model [here](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

- The model is downloaded using the `download_model.sh` script, which uses the `huggingface-cli` to authenticate and download the model files. Set your Hugging Face token in the `.env` file before running the script.

- If you don't want to add your Huggingface token, the application will fall back to using non-gated models.

```
    fallback_models = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2b-it",
    ]
```

## Hosted Example of AI Creative

You can try out a hosted version of AI Creative at [AI Creative Studio](https://huggingface.co/spaces/Zoro-chi/ai-creative-studio). This demo allows you to test the application without needing to set it up locally.

**Note**: The hosted version has limitations and isn't as fast as the local version. For best performance, we recommend running the application locally. This is due to the restrictions from using the free tier of Huggingface Spaces. The local LLM being used here is a less powerful `TinyLlama-1.1B-Chat` model. The total average generation time from text-3D model is **2 Minutes**, while on local its **<1 Minute**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Zoro-chi/ai-creative.git
cd ai-creative
```

### 2. Create a Python virtual environment

```bash
# Using conda
conda create -n ai-creative-py310 python=3.10
conda activate ai-creative-py310

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the LLM model (optional)

To use the local LLM for prompt enhancement:

```bash
bash download_model.sh
```

### 5. Set up environment variables

Create a `.env` file in the project root. Check `.env.example` for the required variables.

## Usage

### Starting the LLM Service

Start the local LLM service first:

```bash
python app/llm/service.py
```

Wait for the model to load (you should see "LLM initialized successfully" in the console).

### Running the UI Application

In a new terminal window:

```bash
python app/ui/app.py
```

This will launch the web interface, accessible at [http://localhost:7860](http://localhost:7860).

### Using the Application

1. **Enter a prompt**: Enter a descriptive text prompt for the 3D model you want to create
2. **Generate**: Click the "Generate" button to start the creation process
3. **View results**: The application will display:
   - The original prompt
   - The enhanced prompt (if LLM enhancement worked)
   - The generated image
   - The 3D model viewer with the created model
   - Download links for both image and 3D model
  


<img width="1470" alt="Image Generation" src="https://github.com/user-attachments/assets/8c06ef61-9db9-4980-b710-fb603dc3a24d" />

Here's a screenshot of the application in action, showing the input prompt, generated image, and the expanded prompt. The total time taken for the generation process is also displayed.

<img width="1470" alt="Model Viewer" src="https://github.com/user-attachments/assets/ffb402a7-5229-412c-b648-161c5323077b" />

_Model Viewer_

<img width="1470" alt="Image Gallery" src="https://github.com/user-attachments/assets/c891a12d-fea6-4237-b743-7820506e10e0" />

_Image Gallery_

<img width="1470" alt="Model Gallery" src="https://github.com/user-attachments/assets/dccaf466-096e-4e17-b9ae-656e5c08be8b" />

_Model Gallery_

## Memory Implementation

AI Creative implements a file-based persistence system for storing generated assets and their metadata. This allows the application to maintain a history of creations and reload them even after application restarts.

### Image Storage

When a text-to-image generation is completed:

1. **Image Files**: Generated images are saved in the `app/data/images/` directory as PNG files
2. **Metadata Files**: For each image, a corresponding JSON metadata file is created with the same base name
3. **Metadata Content**: The metadata includes:
   - Original prompt
   - Enhanced prompt
   - Timestamp
   - Generation parameters
   - Result IDs from Openfabric
   - Download status

Example of image metadata:

```json
{
  "original_prompt": "A crystal palace in the clouds",
  "expanded_prompt": "A majestic crystal palace floating among fluffy white clouds, with sunlight refracting through its transparent walls, casting rainbow patterns. The palace features intricate geometric architecture with tall spires and domed roofs, all made of perfectly clear crystal that sparkles brilliantly.",
  "timestamp": 1744801149,
  "result_id": "data_97531/executions/123456",
  "needs_download": false
}
```

### Model Storage

For 3D model generation:

1. **Model Files**: Generated 3D models are saved in the `app/data/models/` directory as GLB files
2. **Metadata Files**: Each model has an associated JSON metadata file
3. **Metadata Content**: The model metadata includes:
   - Source image reference
   - Model format (always GLB for better compatibility)
   - Timestamp
   - Generation parameters
   - Video preview path (if available)

### Persistence and Loading

- **On-disk Persistence**: All assets and metadata are written to disk immediately after generation
- **Directory Structure**: The application maintains a clean directory structure with separate folders for images and models
- **Auto-discovery**: When the application starts, it scans the data directories to find and load existing assets
- **UI Integration**: Previously generated assets are accessible through the galleries in the UI
- **Download Management**: For remotely stored assets, the application tracks download status and can retrieve them when needed

### Blob Storage Integration

The application can also handle remote assets stored in Openfabric's blob storage:

1. When an image or model is generated, it may initially exist only as a reference to a remote blob
2. The application downloads these blobs as needed and stores them locally
3. The metadata files are updated to reflect the local storage path once downloaded

I couldn't fully implement the instructed memory specs in openfabric_guide.md, but I could if I had more time. The current implementation focuses on a simple file-based approach rather than the more sophisticated database solution outlined in the guide.

## Architecture

AI Creative consists of several key components:

- **Core Pipeline**: Orchestrates the end-to-end creative workflow
- **LLM Service**: Enhances text prompts using a local language model
- **Text-to-Image Service**: Generates images from text (via Openfabric)
- **Image-to-3D Service**: Creates 3D models from images (via Openfabric)
- **Web UI**: Gradio-based user interface

The application uses Openfabric's platform for the compute-intensive image and 3D model generation tasks, while running a lightweight LLM locally for prompt enhancement.

## Logging

AI Creative maintains detailed logs in the following locations:

- `app/core/openfabric_service.log`: Logs for Openfabric service operations
- `app/llm/llm_service.log`: Logs for the local LLM service

## Troubleshooting

### Common Issues

- **LLM Service not starting**: Ensure you have sufficient RAM for the model, and check that the model download was completed successfully
- **Connection errors**: Verify your internet connection and check that your Openfabric app IDs are correct
- **Generation timeout**: Image or 3D model generation may take time, depending on complexity and service load
- **"No module found" errors**: Ensure all dependencies are installed and you're using the correct Python environment
