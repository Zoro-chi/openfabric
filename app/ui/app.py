#!/usr/bin/env python
"""
Minimal Gradio UI Template
"""

import os
import gradio as gr
from pathlib import Path


def main():
    """Simple Gradio interface"""

    # Define a simple demo function
    def greet(name):
        return f"Hello, {name}!"

    # Create a simple interface
    with gr.Blocks(title="Simple Gradio App") as demo:
        gr.Markdown("# Simple Gradio App")

        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(
                    label="Your name",
                    placeholder="Enter your name here...",
                )
                greet_btn = gr.Button("Greet")

            with gr.Column():
                output = gr.Textbox(label="Output")

        # Connect the button to the function
        greet_btn.click(fn=greet, inputs=[name_input], outputs=[output])

    # Launch the UI
    port = int(os.environ.get("UI_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True, debug=True)


if __name__ == "__main__":
    main()
