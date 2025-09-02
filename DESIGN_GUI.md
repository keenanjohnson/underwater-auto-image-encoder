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
1. **Input Support**: GPR images
2. **Output Formats**: TIFF images
3. **Visual Feedback**: Progress bars
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
│  Underwater Image Enhancer                           [X] │
├─────────────────────────────────────────────────────────┤
│  Model:         [/path/to/model.pth]        [Browse]     │
│  Input Folder:  [/path/to/images]           [Browse]     │
│  Output Folder: [/path/to/output]           [Browse]     │
│                                                          │
│  Files Found: 247 GPR images                             │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Progress                                            │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ Processing: image_045.gpr (45/247)                  │ │
│  │ ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░ 18%                  │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Log                                          [Clear]│ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ [12:34:01] Started processing batch                │ │
│  │ [12:34:02] Successfully loaded model                │ │
│  │ [12:34:03] Processing image_044.gpr - Success      │ │
│  │ [12:34:05] Processing image_045.gpr...             │ │
│  │                                                     │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  [Start Processing] [Cancel]                             │
└─────────────────────────────────────────────────────────┘
```
