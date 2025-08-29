# Underwater Image Enhancement GUI Application Design Document

## Executive Summary
A desktop GUI application that enables marine biologists at Seattle Aquarium to locally process underwater ROV survey images using trained ML models. The application wraps the existing inference.py functionality with an intuitive interface designed for non-technical users.

## 1. User Requirements & Use Cases

### Primary Users
- **Marine Biologists**: Non-technical users who need to process underwater images
- **ROV Operators**: Field personnel processing images immediately after collection
- **Research Assistants**: Processing batches of historical survey data

### Core Use Cases

#### UC1: Single Image Processing
- User selects a single GPR/RAW/JPEG image
- Views real-time preview of enhancement
- Saves enhanced image in desired format
- Optional: Compare before/after side-by-side

#### UC2: Batch Processing
- User selects folder containing multiple images
- Sets output directory and format preferences
- Monitors progress with visual feedback
- Reviews results upon completion

#### UC3: Model Management
- User selects from available pre-trained models
- Views model information (training date, performance metrics)
- Downloads new models from repository (optional future feature)

#### UC4: Processing Configuration
- User adjusts processing parameters:
  - Output format (JPEG/TIFF/PNG)
  - Quality settings
  - Processing mode (fast/quality)
  - Denoising options

### Functional Requirements
1. **Input Support**: GPR, DNG, RAW, JPEG, PNG, TIFF formats
2. **Output Formats**: JPEG (default), TIFF, PNG
3. **Processing Modes**:
   - Fast mode: Resized inference (256x256)
   - Quality mode: Full resolution with tiling
   - GPR mode: Includes preprocessing pipeline
4. **Visual Feedback**: Progress bars, image previews, comparison views
5. **Error Handling**: Clear error messages, recovery options
6. **Performance**: Process 4606x4030 image in <30 seconds on standard hardware

### Non-Functional Requirements
- **Platform**: Cross-platform (Windows primary, macOS/Linux secondary)
- **Installation**: Single executable/installer, no Python knowledge required
- **Offline**: Fully functional without internet connection
- **Resource Usage**: <4GB RAM for standard processing
- **UI Response**: Interface remains responsive during processing

## 2. GUI Design & User Flow

### Main Application Layout
```
┌─────────────────────────────────────────────────────────┐
│  [File] [Settings] [Help]                   [Model: v1]  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │                 │  │                 │  [Process]     │
│  │  Input Preview  │→ │ Output Preview  │  [Save]        │
│  │   (Original)    │  │   (Enhanced)    │  [Compare]     │
│  │                 │  │                 │               │
│  └─────────────────┘  └─────────────────┘               │
│                                                          │
│  [Select Image] [Select Folder]          Quality: [▼]    │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Status: Ready                                       │ │
│  │ ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░ 45%                    │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Screen Flows

#### Main Screen
- **Input Section**: Drag-drop area or browse button
- **Preview Panels**: Split view showing original and enhanced
- **Control Panel**: Processing options and action buttons
- **Status Bar**: Progress indicator and messages

#### Batch Processing Screen
```
┌─────────────────────────────────────────────────────────┐
│  Batch Processing                                    [X] │
├─────────────────────────────────────────────────────────┤
│  Input Folder:  [/path/to/images]           [Browse]     │
│  Output Folder: [/path/to/output]           [Browse]     │
│                                                          │
│  Files Found: 247 images (GPR: 200, JPEG: 47)           │
│                                                          │
│  Options:                                                │
│  □ Include subfolders                                    │
│  ☑ Skip existing files                                   │
│  □ Create comparison images                              │
│                                                          │
│  Processing: image_045.gpr (45/247)                      │
│  ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░ 18%                │
│                                                          │
│  [Start Processing] [Pause] [Cancel]                     │
└─────────────────────────────────────────────────────────┘
```

#### Settings Screen
```
┌─────────────────────────────────────────────────────────┐
│  Settings                                            [X] │
├─────────────────────────────────────────────────────────┤
│  Model Selection:                                        │
│  ○ Standard U-Net (v1.0) - Best quality                  │
│  ● Fast U-Net (v1.1) - Faster processing                 │
│  ○ Custom Model [Browse...]                              │
│                                                          │
│  Processing Options:                                     │
│  □ Use GPU acceleration (if available)                   │
│  ☑ Apply denoising post-processing                      │
│  □ Save processing logs                                  │
│                                                          │
│  Output Settings:                                        │
│  Format: [JPEG ▼]  Quality: [95 ▼]                      │
│  Naming: [original_enhanced ▼]                          │
│                                                          │
│  [Save] [Cancel] [Reset Defaults]                        │
└─────────────────────────────────────────────────────────┘
```

## 3. Technical Architecture

### Technology Stack

#### Selected Technologies

- **GUI Framework**: NiceGUI with native desktop mode
- **Packaging**: PyInstaller for standalone executable
- **ML Framework**: PyTorch (existing)
- **Image Processing**: OpenCV, Pillow, rawpy (existing)
- **GPR Processing**: gpr_tools (existing)

#### NiceGUI Framework Details

NiceGUI provides the ideal balance of development speed, modern UI, and desktop functionality:

**Key Features:**
- Native desktop window via `ui.run(native=True)` using pywebview
- Direct file system access through native file dialogs
- Built on FastAPI/Uvicorn for high-performance async operations
- Reactive UI components with automatic updates
- No complex threading required - async/await pattern throughout

**Technical Specifications:**
- Python 3.8+ required
- Dependencies: nicegui, pywebview, fastapi, uvicorn
- Package size: ~100MB with PyInstaller (including pywebview)
- Native mode supports: Windows 10+, macOS 10.10+, Linux (GTK3)

**File Handling Approach:**
```python
from nicegui import app, ui
from pathlib import Path

async def select_and_process():
    # Native file dialog - returns absolute paths
    files = await app.native.main_window.create_file_dialog(
        allow_multiple=True,
        file_types=[('GPR files', '*.gpr'),
                   ('JPEG files', '*.jpg;*.jpeg'),
                   ('All Images', '*.gpr;*.jpg;*.jpeg;*.png;*.tiff')]
    )
    
    for file_path in files:
        # Direct processing - no upload/download needed
        path = Path(file_path)
        if path.suffix.lower() == '.gpr':
            await process_gpr(path)
        else:
            await process_image(path)
```

### NiceGUI Application Architecture

```
┌──────────────────────────────────────────────────────┐
│            NiceGUI Native Application                │
│  ┌──────────────────────────────────────────────┐   │
│  │          PyWebView Native Window             │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │         NiceGUI Components (UI Layer)        │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │   │
│  │  │  Image   │ │ Progress │ │ Settings │    │   │
│  │  │  Viewer  │ │   Card   │ │   Panel  │    │   │
│  │  └──────────┘ └──────────┘ └──────────┘    │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │      FastAPI/Uvicorn Async Backend           │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│              Processing Pipeline                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│
│  │     GPR      │ │   Model      │ │   Image      ││
│  │ Preprocessor │→│  Inference   │→│ Postprocess  ││
│  │  (gpr_tools) │ │  (PyTorch)   │ │  (OpenCV)    ││
│  └──────────────┘ └──────────────┘ └──────────────┘│
└──────────────────────────────────────────────────────┘
```

#### NiceGUI Component Structure

```python
# Main application structure
app/
├── main.py              # Entry point with ui.run(native=True)
├── components/
│   ├── image_viewer.py  # Custom image display component
│   ├── file_selector.py # Native file dialog wrapper
│   └── progress_card.py # Progress tracking component
├── services/
│   ├── inference.py     # Wraps existing Inferencer class
│   ├── preprocessing.py # GPR processing pipeline
│   └── settings.py      # Configuration management
└── utils/
    ├── image_utils.py   # Image format conversions
    └── async_utils.py   # Async helper functions
```

### NiceGUI Component Descriptions

#### UI Components (NiceGUI)
- **Image Viewer**: Split-view component using `ui.image()` for before/after comparison
- **File Selector**: Native dialog wrapper using `app.native.main_window.create_file_dialog()`
- **Progress Card**: Real-time progress using `ui.progress()` and `ui.label()`
- **Settings Panel**: Configuration UI using `ui.select()`, `ui.switch()`, `ui.slider()`
- **Batch Queue**: Table view using `ui.table()` with status indicators

#### Service Layer
- **InferenceService**: Async wrapper around existing `Inferencer` class
- **PreprocessingService**: Async GPR conversion and cropping
- **SettingsService**: JSON-based config persistence
- **ImageService**: Format conversion, caching, thumbnail generation

#### Async Processing Architecture

NiceGUI's async nature eliminates complex threading:

```python
# All processing is async by default
async def process_image(file_path: Path):
    # UI remains responsive during processing
    progress.visible = True
    
    # Preprocessing (if GPR)
    if file_path.suffix.lower() == '.gpr':
        progress.set_text('Converting GPR...')
        dng_path = await preprocess_service.convert_gpr(file_path)
        file_path = dng_path
    
    # Inference
    progress.set_text('Enhancing image...')
    enhanced = await inference_service.process(file_path)
    
    # Update UI - automatic reactivity
    output_image.source = enhanced
    progress.visible = False

# No manual thread management needed
ui.button('Process', on_click=lambda: process_image(selected_file))
```

## 4. Implementation Plan

### Phase 1: Core GUI
- [ ] Set up project structure
- [ ] Implement MainWindow with basic layout
- [ ] Create ImageViewer widget
- [ ] Add file selection/drag-drop
- [ ] Integrate with existing Inferencer class
- [ ] Basic single image processing

### Phase 2: Enhanced Features
- [ ] Batch processing dialog
- [ ] Progress indicators
- [ ] Settings persistence
- [ ] Model selection
- [ ] Comparison view
- [ ] Error handling

### Phase 3: GPR Support
- [ ] Integrate GPR preprocessing
- [ ] Auto-detect file types
- [ ] Processing pipeline selection
- [ ] Format conversion options

### Phase 4: Polish & Packaging
- [ ] UI refinements
- [ ] Performance optimization
- [ ] Create installer/executable
- [ ] Documentation
- [ ] User testing

## 5. File Structure
```
underwater-enhancer-gui/
├── app.py                    # Main entry point
├── components/
│   ├── __init__.py
│   ├── image_viewer.py       # Before/after image display
│   ├── file_selector.py      # Native file dialog integration
│   ├── progress_card.py      # Processing progress display
│   ├── settings_panel.py     # Configuration UI
│   └── batch_queue.py        # Batch processing queue view
├── services/
│   ├── __init__.py
│   ├── inference.py          # Wrapper for existing inference.py
│   ├── preprocessing.py      # GPR conversion service
│   ├── settings.py           # Settings persistence
│   └── image_cache.py        # Image caching service
├── models/                   # Pre-trained model checkpoints
│   └── best_model.pth
├── config/
│   ├── default_settings.json
│   └── user_settings.json   # User preferences (gitignored)
├── static/                   # Static assets for web UI
│   ├── logo.png
│   └── styles.css
├── requirements.txt
├── pyinstaller.spec          # PyInstaller configuration
└── build.py                  # Build script for packaging
```

## 6. Development Priorities

### Must Have (MVP)
1. Single image processing
2. Basic preview functionality
3. Model loading from checkpoint
4. Save enhanced images
5. Simple progress indication

### Should Have
1. Batch processing
2. Before/after comparison
3. Settings persistence
4. Multiple output formats
5. Drag-and-drop support

### Nice to Have
1. GPR preprocessing integration
2. Model comparison tool
3. Processing history
4. Export statistics
5. Cloud model repository

## 7. Risk Mitigation

### Technical Risks
- **Large Image Memory**: Implement tiled processing (already in inference.py)
- **GPU Availability**: Fallback to CPU processing
- **Model Compatibility**: Version checking, error messages
- **Package Size**: Optimize dependencies, separate model downloads

### User Experience Risks
- **Processing Time**: Clear progress indicators, time estimates
- **File Format Issues**: Comprehensive format validation
- **System Resources**: Memory monitoring, processing limits

## 8. Testing Strategy

### Unit Tests
- Image loading/saving
- Model inference
- Configuration management

### Integration Tests
- Full processing pipeline
- Batch processing
- Error handling

### User Acceptance Tests
- Marine biologist workflow validation
- Performance benchmarks
- Usability testing

## 9. Deployment with PyInstaller

### PyInstaller Configuration

```python
# pyinstaller.spec
a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[
        ('gpr_tools', '.'),  # Include gpr_tools binary
    ],
    datas=[
        ('models/*.pth', 'models'),
        ('config/default_settings.json', 'config'),
        ('static/*', 'static'),
    ],
    hiddenimports=[
        'nicegui',
        'pywebview',
        'fastapi',
        'uvicorn',
        'torch',
        'torchvision',
        'PIL',
        'cv2',
        'rawpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='UnderwaterEnhancer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='static/logo.ico'
)
```

### Build Process

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
pyinstaller pyinstaller.spec

# Output location
dist/UnderwaterEnhancer.exe  # Windows
dist/UnderwaterEnhancer      # Linux/macOS
```

### Platform-Specific Considerations

**Windows:**
- Include Visual C++ Redistributables
- Sign executable to avoid SmartScreen warnings
- Expected size: ~250MB

**macOS:**
- Code signing required for distribution
- Create .dmg installer
- Expected size: ~300MB

**Linux:**
- May need to bundle GTK3 libraries
- Consider AppImage format
- Expected size: ~280MB

### Distribution Strategy

1. **GitHub Releases**
   - Automated builds via GitHub Actions
   - Platform-specific installers
   - Version changelog

2. **Internal Deployment**
   - Network share distribution
   - Include model files separately (optional)
   - Update notification system

## 10. Next Steps

1. **Validate Requirements**: Review with marine biology team
2. **Prototype UI**: Create mockups for user feedback
3. **Select Framework**: Final decision on GUI technology
4. **Begin Implementation**: Start with Phase 1 core functionality