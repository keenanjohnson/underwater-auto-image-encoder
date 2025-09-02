# Underwater Image Enhancement GUI Application Design Document

## Executive Summary
A desktop GUI application that enables marine biologists at Seattle Aquarium to locally process underwater ROV survey images using trained ML models. The application wraps the existing inference.py functionality with an intuitive interface designed for non-technical users.

## 1. User Requirements & Use Cases

### Primary Users
- **Marine Biologists**: Non-technical users who need to process underwater images
- **ROV Operators**: Field personnel processing images immediately after collection
- **Research Assistants**: Processing batches of historical survey data

### Core Use Case

#### UC1: Single Image Processing
- User selects a model from their local machine
- User selects one or more GPR images
- Saves enhanced image in .tiff format
- Optional: Compare before/after side-by-side

### Functional Requirements
1. **Input Support**: GPR images (native support via bundled gpr_tools)
2. **Output Formats**: TIFF or JPEG
3. **Visual Feedback**: Progress bars
4. **Batch Processing**: Process multiple images with queue management
5. **Error Handling**: Clear error messages, recovery options
6. **Performance**: Process 4606x4030 image in <30 seconds on standard hardware

### Non-Functional Requirements
- **Platform**: Cross-platform (Windows primary, macOS/Linux secondary)
- **Installation**: Single executable, no Python knowledge required
- **Offline**: Fully functional without internet connection
- **UI Response**: Interface remains responsive during processing

## 2. GUI Design Mockup

```
┌─────────────────────────────────────────────────────────┐
│  🌊 Underwater Image Enhancer                    [−][□][X]│
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Model Selection                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ [/path/to/model.pth]                    [Browse...]  │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  Input/Output Folders                                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Input:  [/path/to/images]              [Browse...]  │ │
│  │ Output: [/path/to/output]              [Browse...]  │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  Files Found: 247 GPR images                             │
│  Format: ☑ TIFF  ☐ JPEG                                 │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Progress                                            │ │
│  │ Processing: image_045.gpr (45/247)                  │ │
│  │ ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░ 18%                  │ │
│  │ Time Elapsed: 00:12:34 | Est. Remaining: 00:55:20  │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Log                                          [Clear]│ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ [12:34:01] Started processing batch                │ │
│  │ [12:34:02] Successfully loaded model                │ │
│  │ [12:34:03] ✓ image_044.gpr processed               │ │
│  │ [12:34:05] Processing image_045.gpr...             │ │
│  │                                                     │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  [▶ Start Processing]  [⏸ Pause]  [■ Cancel]            │
│                                                          │
│  [🌙 Dark Mode]                              [Settings ⚙]│
└─────────────────────────────────────────────────────────┘
```

### Design Notes:
- **CustomTkinter Benefits**: Modern rounded corners, smooth animations, native dark/light mode
- **Visual Hierarchy**: Grouped sections with frames for better organization
- **User Feedback**: Time estimates and clearer success indicators
- **Professional Look**: Consistent with modern desktop applications

## 3. Technical Architecture

### GUI Framework Choice: CustomTkinter

**Why CustomTkinter:**
- **Modern UI**: Professional appearance with rounded corners, hover effects, and smooth animations
- **Dark/Light Mode**: Built-in theme switching that follows system preferences
- **Small Footprint**: Based on Tkinter (included with Python), adds only ~10MB to executable
- **PyInstaller Compatible**: Excellent packaging support with minimal configuration
- **Cross-Platform**: Consistent look across Windows, macOS, and Linux
- **No Licensing Issues**: MIT license allows commercial use

**Key Features for This Project:**
- Native file dialogs via tkinter.filedialog
- CTkProgressBar for smooth progress indication
- CTkTextbox for scrollable log output
- CTkButton with hover states for better UX
- CTkFrame for visual grouping of controls
- Threading support for responsive UI during processing

### GPR File Support Strategy
The application will provide native GPR support by bundling the `gpr_tools` binary directly within the executable. This eliminates external dependencies while maintaining full GPR processing capabilities.

**Implementation approach:**
- Bundle platform-specific gpr_tools binaries (Windows/macOS/Linux)
- Automatic platform detection and binary selection at runtime
- Transparent GPR→TIFF conversion in memory before ML processing
- No temporary files or user-visible conversion steps

For detailed implementation plan, see: `GUI_GPR_BUNDLING_PLAN.md`

### Core Components
1. **GPR Converter Module** (`src/converters/gpr_converter.py`)
   - Wraps bundled gpr_tools binary
   - Handles GPR→TIFF conversion seamlessly
   - Platform-agnostic interface

2. **Image Processor** (`src/gui/image_processor.py`)
   - Integrates GPR converter with ML inference pipeline
   - Handles batch processing with progress tracking
   - Memory-efficient processing for large images

3. **GUI Application** (`app.py`)
   - CustomTkinter-based interface
   - Native file dialogs via tkinter.filedialog
   - Real-time progress updates with CTkProgressBar
   - Threading for background processing to maintain UI responsiveness

### Packaging & Distribution
- **PyInstaller** for single-executable packaging
- **Size**: ~275MB (includes PyTorch, ML model, and gpr_tools)
- **No installation required**: Users download and run single file
- **Cross-platform**: Same codebase, platform-specific builds
