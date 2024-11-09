<div align="center">
  <img src="frontend/react-app/src/assets/Upscalee_logo.png" alt="Logo" width="200">
  <img src="frontend/react-app/src/assets/upscalee.png" alt="Logo" width="400">

  # Upscalee
  
  An AI-powered image upscaling application with real-time quality assessment
  
  ![GitHub repo size](https://img.shields.io/github/repo-size/DieterR97/Upscalee?color=000000)
  ![GitHub language count](https://img.shields.io/github/languages/count/DieterR97/Upscalee?color=000000)
  ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/DieterR97/Upscalee?color=000000)

  [View Demo](#) · [Report Bug](https://github.com/DieterR97/Upscalee/issues) · [Request Feature](https://github.com/DieterR97/Upscalee/issues)
</div>

## Table of Contents
* [About the Project](#about-the-project)
  * [Project Description](#project-description)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Features and Functionality](#features-and-functionality)
  * [AI-Powered Upscaling](#-ai-powered-upscaling)
  * [Real-time Comparison](#-real-time-comparison)
  * [Quality Assessment](#-quality-assessment)
  * [Advanced Configuration](#-advanced-configuration)
* [Concept Process](#concept-process)
  * [Ideation](#ideation)
  * [User-flow](#user-flow)
* [Development Process](#development-process)
  * [Implementation Process](#implementation-process)
  * [Future Implementation](#future-implementation)
  * [Highlights](#highlights)
  * [Challenges](#challenges)
* [Final Outcome](#final-outcome)
  * [Mockups](#mockups)
  * [Video Demonstration](#video-demonstration)
  * [Highlights](#highlights-1)
  * [Challenges](#challenges-1)
  * [Reviews & Testing](#reviews--testing)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About the Project

### Project Description
Upscalee is a powerful image upscaling application that leverages state-of-the-art AI models to enhance image resolution while maintaining quality. The application features a user-friendly interface with real-time comparison tools and comprehensive image quality assessment capabilities.

<div align="center">
  <!-- Add a screenshot or demo GIF of your application here -->
  <img src="path/to/screenshot.png" alt="Project Screenshot">
</div>

### Built With
* **Frontend:**
  * React
  * TypeScript
  * Material-UI
  * Axios
* **Backend:**
  * Flask
  * PyTorch
  * Real-ESRGAN
  * OpenCV
  * Pillow
  * PyIQA

## Getting Started

### Prerequisites
* Python 3.10 or higher
* Node.js and npm
* CUDA-capable GPU (optional, but recommended for better performance)

### Installation

1. Clone the repository

```bash
git clone https://github.com/DieterR97/Upscalee.git
cd Upscalee
```
2. Run the setup script

```bash
setup.bat
```
This will:
- Set up the Python virtual environment
- Install all required Python packages
- Install frontend dependencies
- Create necessary directories
- Configure CUDA if available

3. Start the application

```bash
run.bat
```
Or manually:
- Activate virtual environment: `venv\Scripts\activate`
- Start frontend: `cd frontend/react-app & npm start`
- Start backend: `cd backend & python app.py`

## Features and Functionality

### 🤖 AI-Powered Upscaling
* Multiple AI models including Real-ESRGAN and custom models
* Support for various upscaling factors (2x, 4x)
* CUDA acceleration for faster processing

### ⚖️ Real-time Comparison
* Interactive slider for before/after comparison
* Side-by-side view of original and upscaled images
* Detailed image information and metadata

### 📊 Quality Assessment
* Comprehensive image quality metrics
* Both reference and no-reference quality assessment
* Real-time quality score calculation

### ⚙️ Advanced Configuration
* Custom model support
* Configurable processing parameters
* GPU/CPU processing options

## Concept Process

### Ideation
<!-- Add your ideation process, sketches, or brainstorming results -->

### User-flow
<!-- Add a user flow diagram showing how users interact with your application -->

## Development Process

### Implementation Process
* **Frontend Architecture:**
  * React with TypeScript for type safety
  * Component-based design for modularity
  * Material-UI for consistent styling

* **Backend Architecture:**
  * Flask REST API
  * PyTorch for AI model inference
  * Efficient image processing pipeline

### Future Implementation
* Additional AI models support
* Batch processing capabilities
* Advanced image preprocessing options
* User accounts and image history
* API endpoint documentation

### Highlights
* Successfully implemented Real-ESRGAN model integration
* Achieved real-time image quality assessment
* Created an intuitive comparison interface

### Challenges
* Optimizing GPU memory usage for large images
* Implementing efficient image processing pipeline
* Balancing quality and processing speed

## Final Outcome

### Mockups
<!-- Add final application screenshots/mockups -->

### Video Demonstration
<!-- Add a link to your demo video if available -->
[View Demonstration](#)

### Highlights
* Successfully implemented Real-ESRGAN model integration
* Achieved real-time image quality assessment
* Created an intuitive comparison interface

### Challenges
* Optimizing GPU memory usage for large images
* Implementing efficient image processing pipeline
* Balancing quality and processing speed

### Reviews & Testing
<!-- Add information about any testing performed -->
[View Demonstration](#)

## Contributing
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Dieter Roelofse - [GitHub](https://github.com/DieterR97)

Project Link: [https://github.com/DieterR97/Upscalee](https://github.com/DieterR97/Upscalee)

## Acknowledgements
* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
* [PyIQA](https://github.com/chaofengc/IQA-PyTorch)
* [React](https://reactjs.org/)
* [Flask](https://flask.palletsprojects.com/)
* [PyTorch](https://pytorch.org/)


