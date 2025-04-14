#!/usr/bin/env python3
"""
Openfabric Blob Viewer - A utility for viewing and downloading resources from Openfabric

Usage:
  python blob_viewer.py view <data_blob_id> [<execution_id>]
  python blob_viewer.py download <data_blob_id> [<execution_id>]

Examples:
  # View an image directly in browser
  python blob_viewer.py view data_blob_1d9d210d20c1e75ea6a3855b6d10341fd8f125b49866b61b7ae94f8fa4bffd49 2d529306be574949a2a3d2f9d9e4082b

  # Download a resource
  python blob_viewer.py download data_blob_1d9d210d20c1e75ea6a3855b6d10341fd8f125b49866b61b7ae94f8fa4bffd49 2d529306be574949a2a3d2f9d9e4082b
"""

import os
import sys
import argparse
import webbrowser
import requests
from pathlib import Path
from dotenv import load_dotenv
import base64
import json
from datetime import datetime

# Make sure we can import from our app
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Configure output directory for downloads
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "downloads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get app IDs from environment
TEXT_TO_IMAGE_APP_ID = os.environ.get("TEXT_TO_IMAGE_APP_ID")
IMAGE_TO_3D_APP_ID = os.environ.get("IMAGE_TO_3D_APP_ID")


def construct_resource_url(data_blob_id, execution_id=None):
    """
    Construct the URL for accessing a resource from a data_blob ID

    Args:
        data_blob_id: The data_blob ID (can be full path or just the ID)
        execution_id: Optional execution ID

    Returns:
        URL to access the resource
    """
    # Extract the actual blob ID if provided with path format
    if "/" in data_blob_id:
        parts = data_blob_id.split("/")
        data_blob_id = parts[0]
        if len(parts) > 2 and not execution_id:
            execution_id = parts[2]

    # Create the reid parameter value
    reid = data_blob_id
    if execution_id:
        reid = f"{data_blob_id}/executions/{execution_id}"

    # Format the URL correctly based on the example
    base_url = f"https://{TEXT_TO_IMAGE_APP_ID}/resource?reid={reid}"

    return base_url


def open_in_browser(data_blob_id, execution_id=None):
    """Open a resource directly in the web browser"""
    url = construct_resource_url(data_blob_id, execution_id)
    print(f"Opening URL in browser: {url}")
    webbrowser.open(url)


def download_resource(
    data_blob_id, execution_id=None, prompt=None, target_dir=None, metadata=None
):
    """
    Download a resource from the given data_blob ID

    Args:
        data_blob_id: The data_blob ID
        execution_id: Optional execution ID
        prompt: Optional prompt text to use in filename
        target_dir: Optional target directory to save to (defaults to downloads)
        metadata: Optional metadata to save alongside the downloaded file
    """
    url = construct_resource_url(data_blob_id, execution_id)

    # Use downloads directory as default if not specified
    if target_dir:
        output_dir = Path(target_dir)
    else:
        output_dir = OUTPUT_DIR

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading from: {url}")
        response = requests.get(url)

        if response.status_code == 200:
            # Determine content type and extension
            content_type = response.headers.get(
                "Content-Type", "application/octet-stream"
            )

            # Choose file extension based on content type
            extension = "bin"  # Default extension
            if "image/png" in content_type:
                extension = "png"
            elif "image/jpeg" in content_type:
                extension = "jpg"
            elif (
                "model/gltf+json" in content_type or "application/json" in content_type
            ):
                extension = "gltf"
            elif "model/gltf-binary" in content_type:
                extension = "glb"

            # Create filename based on prompt if available
            if prompt:
                # Use first 15 chars of prompt, replacing spaces with underscores
                base_name = prompt[:15].strip().replace(" ", "_").replace("/", "_")
                # Remove any other non-alphanumeric characters
                base_name = "".join(c for c in base_name if c.isalnum() or c == "_")

                # Add timestamp for uniqueness
                timestamp = int(datetime.now().timestamp())
                filename = f"{base_name}_{timestamp}.{extension}"

                # Also create metadata filename
                metadata_filename = f"{base_name}_{timestamp}.json"
            else:
                # Fallback to timestamp and blob ID if no prompt
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_blob_id = data_blob_id.replace("/", "_")

                if execution_id:
                    filename = f"{timestamp}_{clean_blob_id[:8]}_{execution_id[:8]}.{extension}"
                else:
                    filename = f"{timestamp}_{clean_blob_id[:8]}.{extension}"

                # Also create metadata filename
                metadata_filename = filename.replace(f".{extension}", ".json")

            output_path = output_dir / filename
            metadata_path = output_dir / metadata_filename

            # Save the file
            with open(output_path, "wb") as f:
                f.write(response.content)

            # Create and save metadata
            if metadata:
                metadata["download_timestamp"] = int(datetime.now().timestamp())
                metadata["download_source"] = url
                metadata["file_path"] = str(output_path)

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            print(f"Successfully downloaded to: {output_path}")
            return str(output_path)
        else:
            print(f"Failed to download resource. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"Error downloading resource: {str(e)}")
        return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Openfabric Blob Viewer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # View command
    view_parser = subparsers.add_parser("view", help="View a blob in browser")
    view_parser.add_argument("data_blob_id", help="Blob ID or full path")
    view_parser.add_argument("execution_id", nargs="?", help="Execution ID (optional)")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a blob")
    download_parser.add_argument("data_blob_id", help="Blob ID or full path")
    download_parser.add_argument(
        "execution_id", nargs="?", help="Execution ID (optional)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.command:
        print(__doc__)
        return

    if args.command == "view":
        open_in_browser(args.data_blob_id, args.execution_id)
    elif args.command == "download":
        download_resource(args.data_blob_id, args.execution_id)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
