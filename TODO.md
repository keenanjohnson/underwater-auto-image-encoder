This is the project todo list.

## ML Pipeline Development
[x] Set up the local development environment using vscode dev containers to keep dependencies isolated and ensure a consistent development environment across different machines.
[x] Create a script for automating the preprocessing steps, including GPR to RAW conversion and image cropping.
[x] Select the ML framework (e.g., TensorFlow, PyTorch) for building the autoencoder.
[x] Design the autoencoder model architecture.
[x] Implement the autoencoder model architecture using the selected machine learning framework.
[x] Train the autoencoder model on the preprocessed dataset.
[] Evaluate the model's performance and make necessary adjustments.
[] Integrate the trained model into the output processing pipeline.

## GUI Application Development (See DESIGN_GUI.md for details)

### Phase 1: Core GUI
[] Set up NiceGUI project structure
[] Implement main window with basic layout
[] Create image viewer component with before/after split view
[] Add native file selection dialog integration
[] Integrate with existing Inferencer class
[] Implement basic single image processing

### Phase 2: Enhanced Features
[] Implement batch processing with queue view
[] Add real-time progress indicators
[] Create settings panel with persistence
[] Add model selection dropdown
[] Implement comparison view toggle
[] Add comprehensive error handling

### Phase 3: GPR Support
[] Integrate GPR preprocessing pipeline
[] Add auto-detection for file types
[] Implement processing pipeline selection
[] Add format conversion options

### Phase 4: Polish & Packaging
[] Refine UI/UX based on feedback
[] Optimize performance for large images
[] Configure PyInstaller spec file
[] Create platform-specific installers
[] Write user documentation
[] Conduct user testing with marine biologists
