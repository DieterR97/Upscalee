import React, { useState, useRef, useEffect } from 'react';
import './style.css';
import ImageComparisonSlider from './components/ImageComparisonSlider/ImageComparisonSlider';
import ImageQualityAssessment from './components/ImageQualityAssessment/ImageQualityAssessment';
import ImageQueue from './components/ImageQueue/ImageQueue';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import DropZone from './components/DropZone/DropZone';
import logo from './assets/upscalee.png';
import logo2 from './assets/Upscalee_logo.png';
import { Tabs, Box } from '@mui/material';
import reactLogo from './assets/react.svg';
import pythonLogo from './assets/python.svg';
import pytorchLogo from './assets/pytorch.svg';
import flaskLogo from './assets/flask.png';
import cat from './assets/cat.png';
import IQA_PyTorch from './assets/pyiqa.png';
import OpenCV from './assets/opencv.svg';
import ee from './assets/ee.png';
import catppuccin from './assets/catppuccin.png';

// Define interfaces for type safety and better code documentation
interface ModelInfo {
  description: string;  // Description of the upscaling model
  scale: number;        // Default scale factor for the model
  variable_scale: boolean;  // Whether the model supports different scale factors
  is_spandrel?: boolean;   // Whether this is a Spandrel model (optional)
  spandrel_info?: {        // Spandrel-specific information (optional)
    input_channels: number;
    output_channels: number;
    supports_half: boolean;
    supports_bfloat16: boolean;
    size_requirements: any;
    tiling: string;
    scale: number;
    framework?: string;
  };
  name?: string;           // Display name for the model (optional)
  architecture?: string;   // Model architecture (optional)
  file_pattern?: string;   // File pattern for custom models (optional)
  source_url?: string;    // Add this line
}

interface ModelOptions {
  [key: string]: ModelInfo;
}

interface QueuedImageInfo {
  modelName: string;
  scale: number;
  originalImage: string;
  upscaledImage: string;
}

interface TabProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
}

// Add these interfaces near the top
interface Config {
  maxImageDimension: number;
  modelPath: string;
}

// Add these interfaces
interface UnregisteredModel {
  name: string;
  file_pattern: string;
  scale: number | null;
  path: string;
}

interface ModelScanResult {
  registered: ModelOptions;
  unregistered: UnregisteredModel[];
  message?: string;
  removed?: string[];
}

// Add this interface at the top of your file
interface Window {
  showDirectoryPicker(): Promise<any>;
}

/**
 * Tab component for navigation between different sections of the application
 * Handles visual styling and click events for tab selection
 */
const Tab: React.FC<TabProps> = ({ label, isActive, onClick }) => (
  <button
    className={`tab ${isActive ? 'active' : ''}`}
    onClick={onClick}
  >
    {label}
  </button>
);

/**
 * Main component for the image upscaling application
 * Manages image processing, model selection, and UI state
 */
const ImageUpscaler: React.FC = () => {
  // State Management Section
  // Core image processing states
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [upscaledImage, setUpscaledImage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Model and configuration state
  const [models, setModels] = useState<ModelOptions>({});
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedScale, setSelectedScale] = useState<number>(4);

  // UI state management
  const [showQueue, setShowQueue] = useState(false);
  const [queuedImages, setQueuedImages] = useState<QueuedImageInfo[]>([]);
  const [activeTab, setActiveTab] = useState<'about' | 'upscale' | 'model-info' | 'info' | 'config'>('about');
  const [imageInfo, setImageInfo] = useState<any>(null);

  // Download progress tracking
  const [modelDownloading, setModelDownloading] = useState<boolean>(false);
  const [downloadProgress, setDownloadProgress] = useState<string>('');

  // Add these state variables in the ImageUpscaler component
  const [config, setConfig] = useState<Config>({
    maxImageDimension: 1024,
    modelPath: ''
  });
  const [isConfigChanged, setIsConfigChanged] = useState(false);

  // Add these states to the ImageUpscaler component
  const [unregisteredModels, setUnregisteredModels] = useState<UnregisteredModel[]>([]);
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [selectedUnregisteredModel, setSelectedUnregisteredModel] = useState<UnregisteredModel | null>(null);

  // Add this state near your other state declarations
  const [showWelcomeModal, setShowWelcomeModal] = useState(true);

  /**
   * Fetches available upscaling models from the backend when component mounts
   * Updates the models state with retrieved data
   */
  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/models');
      if (response.ok) {
        const modelData: ModelScanResult = await response.json();
        setModels(modelData.registered);
        setUnregisteredModels(modelData.unregistered);

        // Show notification for removed models if any
        if (modelData.message) {
          toast.warning(modelData.message, {
            position: "top-center",
            autoClose: 5000,
            toastId: 'removed-models',
          });
        }

        // Show notification for unregistered models if any
        if (modelData.unregistered.length > 0) {
          const count = modelData.unregistered.length;
          toast.info(
            <>
              Found {count} unregistered model{count > 1 ? 's' : ''}.
              Go to the Configuration tab to register {count > 1 ? 'them' : 'it'}.
            </>,
            {
              position: "top-center",
              autoClose: 5000,
              toastId: 'unregistered-models',
            }
          );
        }
      }
    } catch (error) {
      console.error('Error fetching models:', error);
      toast.error('Failed to fetch models');
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  /**
   * Handles cleanup and state reset when switching between tabs
   * Ensures proper UI state management during tab navigation
   */
  useEffect(() => {
    if (activeTab === 'info') {
      // Clear all image-related state when switching to info tab
      setImageInfo(null);
      setSelectedImage(null);
      setImagePreview(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // Reset file input
      }
    } else if (activeTab === 'upscale') {
      // Clear queue-related state when switching to upscale tab
      setShowQueue(false);
      setQueuedImages([]);
      setImagePreview(null);
    }
  }, [activeTab]);

  // Handle image selection
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setUpscaledImage(null);
    }
  };

  // Check model status
  const checkModelStatus = async (modelName: string) => {
    try {
      const response = await fetch(`http://localhost:5000/model-status/${modelName}`);
      if (response.ok) {
        const data = await response.json();
        return data.downloaded;
      }
      return false;
    } catch (error) {
      console.error('Error checking model status:', error);
      return false;
    }
  };

  /**
   * Processes the image upscaling request
   * Handles model downloading, image processing, and error states
   * Updates UI with progress and results
   */
  const upscaleImage = async () => {
    if (!selectedImage) return;

    // Clear the upscaled image before starting the new upscale
    setUpscaledImage(null);

    // Scroll to the placeholder container immediately after clicking
    const placeholderContainer = document.querySelector('.placeholder-container');
    if (placeholderContainer) {
      placeholderContainer.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }

    // Check if the selected model is a Spandrel model
    const isSpandrelModel = selectedModel in models && models[selectedModel].is_spandrel;

    // Only check model download status for non-Spandrel models
    if (!isSpandrelModel) {
      // Check if model is downloaded
      const isModelDownloaded = await checkModelStatus(selectedModel);
      if (!isModelDownloaded) {
        setModelDownloading(true);
        setDownloadProgress('Downloading model weights... This may take a few minutes. This is a one-time download for future use.');
      }
    }

    const formData = new FormData();
    formData.append('image', selectedImage);
    formData.append('model', selectedModel);
    formData.append('scale', selectedScale.toString());

    setLoading(true); // Set loading to true before starting the upscale process

    try {
      // Send image to backend for upscaling
      const response = await fetch('http://localhost:5000/upscale', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const upscaledImageURL = URL.createObjectURL(blob);
        setLoading(false); // Set loading to false after the process is done
        setUpscaledImage(upscaledImageURL);

        // Add success toast with quality assessment button
        toast.success(
          <div>
            <p>Image successfully upscaled using {selectedModel} ({selectedScale}x)</p>
            <button
              onClick={() => {
                const qualitySection = document.querySelector('.iqa-container');
                if (qualitySection) {
                  qualitySection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
              }}
              style={{
                background: 'white',
                color: '#333',
                border: '1px solid #ccc',
                padding: '5px 10px',
                borderRadius: '4px',
                marginTop: '8px',
                cursor: 'pointer',
                width: '100%'
              }}
            >
              View Quality Assessment ↓
            </button>
          </div>,
          {
            position: "top-center",
            autoClose: 5000,
          }
        );

        // Add small delay to ensure components are rendered
        setTimeout(() => {
          const comparisonTitle = document.querySelector('.App h1');
          if (comparisonTitle) {
            comparisonTitle.scrollIntoView({
              behavior: 'smooth',
              block: 'start'
            });
          }
        }, 100);

      } else {
        toast.error('Failed to upscale image. Please try again.', {
          position: "top-center",
          autoClose: 3000,
        });
        console.error('Upscaling failed');
      }
    } catch (error) {
      toast.error('Error occurred while upscaling. Please try again.', {
        position: "top-center",
        autoClose: 3000,
      });
      console.error('Error upscaling image:', error);
    } finally {
      setLoading(false);
      setModelDownloading(false);
      setDownloadProgress('');
    }
  };

  // Reset the input and clear selected images
  const resetInput = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setUpscaledImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // Reset file input
    }
  };

  /**
   * Manages the scale selector UI element
   * Only displays for models that support variable scaling
   */
  const showScaleSelector = () => {
    if (!models[selectedModel]?.variable_scale) return null;

    return (
      <div className="scale-selection">
        <label htmlFor="scale-select">Select Scale: </label>
        <select
          id="model-select"
          value={selectedScale}
          onChange={(e) => setSelectedScale(Number(e.target.value))}
          disabled={loading}
        >
          <option value="1">x1</option>
          <option value="2">x2</option>
          <option value="3">x3</option>
          <option value="4">x4</option>
        </select>
      </div>
    );
  };

  useEffect(() => {
    if (models[selectedModel]) {
      setSelectedScale(models[selectedModel].scale);
    }
  }, [selectedModel, models]);

  /**
   * Adds the current image comparison to the queue
   * Displays notification and updates queue state
   */
  const handleQueue = () => {
    if (upscaledImage && imagePreview) {
      const newQueuedImage: QueuedImageInfo = {
        modelName: selectedModel,
        scale: selectedScale,
        originalImage: imagePreview,
        upscaledImage: upscaledImage
      };

      setQueuedImages(prev => [...prev, newQueuedImage]);
      setShowQueue(true);

      console.log('Queue button clicked, showing toast');

      toast.info('Image added to the queue! Scroll down to view the queue.', {
        position: "top-center",
        autoClose: 4000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
      });
    }
  };

  /**
   * Retrieves detailed information about the selected image
   * Updates UI with image metadata and properties
   */
  const fetchImageInfo = async (file: File) => {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:5000/image-info', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const info = await response.json();
        setImageInfo(info);

        // Add success toast notification
        toast.success(`Successfully retrieved information for ${file.name}`, {
          position: "top-center",
          autoClose: 4000,
          hideProgressBar: false,
          closeOnClick: true,
          pauseOnHover: true,
          draggable: true,
          progress: undefined,
        });
      } else {
        toast.error('Failed to fetch image information. Please try again.', {
          position: "top-center",
          autoClose: 3000,
        });
      }
    } catch (error) {
      toast.error('Error occurred while fetching image information. Please try again.', {
        position: "top-center",
        autoClose: 3000,
      });
      console.error('Error fetching image info:', error);
    }
  };

  // Add this effect to load the initial configuration
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('http://localhost:5000/config');
        if (response.ok) {
          const configData = await response.json();
          setConfig(configData);
        }
      } catch (error) {
        console.error('Error fetching configuration:', error);
        toast.error('Failed to load configuration');
      }
    };

    fetchConfig();
  }, []);

  // Add this function to handle configuration updates
  const handleSaveConfig = async () => {
    try {
      const response = await fetch('http://localhost:5000/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        setIsConfigChanged(false);
        toast.success('Configuration saved successfully');
      } else {
        toast.error('Failed to save configuration');
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
      toast.error('Error saving configuration');
    }
  };

  // Add the ModelRegistrationForm component
  const ModelRegistrationForm: React.FC<{
    model: any;
    onClose: () => void;
    onRegister: () => void;
  }> = ({ model, onClose, onRegister }) => {
    const [formData, setFormData] = useState({
      name: model.name,
      display_name: model.name,
      description: "",
      scale: model.scale || 4,
      variable_scale: false,
      architecture: "",
      source_url: "",
      file_pattern: model.file_pattern
    });

    const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault();
      try {
        const response = await fetch('http://localhost:5000/register-model', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData)
        });

        if (response.ok) {
          toast.success('Model registered successfully');
          onRegister();
          onClose();
        } else {
          const error = await response.json();
          toast.error(`Failed to register model: ${error.error}`);
        }
      } catch (error) {
        toast.error('Failed to register model');
        console.error('Error registering model:', error);
      }
    };

    // Add click outside handler
    const handleClickOutside = (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) {
        onClose();
      }
    };

    return (
      <div className="modal" onClick={handleClickOutside}>
        <div className="modal-content">
          <h2>Register New Model</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Display Name:</label>
              <input
                type="text"
                value={formData.display_name}
                onChange={e => setFormData(prev => ({ ...prev, display_name: e.target.value }))}
                required
              />
            </div>
            <div className="form-group">
              <label>Description:</label>
              <textarea
                value={formData.description}
                onChange={e => setFormData(prev => ({ ...prev, description: e.target.value }))}
                required
              />
            </div>
            <div className="form-group">
              <label>Scale:</label>
              <input
                type="number"
                value={formData.scale}
                onChange={e => setFormData(prev => ({ ...prev, scale: parseInt(e.target.value) }))}
                required
                min={1}
                max={8}
              />
            </div>
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={formData.variable_scale}
                  onChange={e => setFormData(prev => ({ ...prev, variable_scale: e.target.checked }))}
                />
                Variable Scale
              </label>
            </div>
            <div className="form-group">
              <label>Architecture:</label>
              <input
                type="text"
                value={formData.architecture}
                onChange={e => setFormData(prev => ({ ...prev, architecture: e.target.value }))}
                required
                placeholder="e.g., Real-ESRGAN"
              />
            </div>
            <div className="form-group">
              <label>Source URL:</label>
              <input
                type="url"
                value={formData.source_url}
                className="input-url"
                onChange={e => setFormData(prev => ({ ...prev, source_url: e.target.value }))}
                placeholder="e.g., https://github.com/username/repo"
              />
            </div>
            <div className="button-group register-model-buttons">
              <button type="button" onClick={onClose} className='reset-button'>Cancel</button>
              <button type="submit">Register Model</button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  const handleFileSelect = (file: File) => {
    setSelectedImage(file);
    setImagePreview(URL.createObjectURL(file));
    setUpscaledImage(null);
  };

  // Add this component inside your App.tsx but before the main component
  const WelcomeModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
    return (
      <div className="modal welcome-modal">
        <div className="modal-content welcome-content">
          <h2>Welcome! 👋</h2>
          <p>
            This application features helpful tooltips throughout the interface.
            If you're unsure about any feature or want to learn more, simply
            hover your mouse over the element to see a detailed explanation.
          </p>
          <div className="tooltip-example">
            <span className="example-element" data-tooltip="Like this!">Hover over me</span>
          </div>
          <button
            className="close-welcome-button"
            onClick={onClose}
          >
            Got it!
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="Root">

      {showWelcomeModal && (
        <WelcomeModal onClose={() => setShowWelcomeModal(false)} />
      )}
      <ToastContainer
        position="top-center"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />

      <div className='nav-container'>

        {/* <h1>Image Upscaler</h1> */}
        <div className="logo-container">
          <img src={logo2} alt="Logo" className="logo" />
          <img src={logo} alt="Logo" className="logoName" />
        </div>

        {/* Add the tab navigation */}
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'about' ? 'active' : ''}`}
            onClick={() => setActiveTab('about')}
            data-tooltip="Learn about the application and its features"
          >
            About
          </button>
          <button
            className={`tab ${activeTab === 'upscale' ? 'active' : ''}`}
            onClick={() => setActiveTab('upscale')}
            data-tooltip="Upscale your images using AI models"
          >
            Upscale
          </button>
          <button
            className={`tab ${activeTab === 'model-info' ? 'active' : ''}`}
            onClick={() => setActiveTab('model-info')}
            data-tooltip="View and select from available upscaling models"
          >
            Models
          </button>
          <button
            className={`tab ${activeTab === 'info' ? 'active' : ''}`}
            onClick={() => setActiveTab('info')}
            data-tooltip="Analyze image properties and metadata"
          >
            Image Info
          </button>
          <button
            className={`tab ${activeTab === 'config' ? 'active' : ''}`}
            onClick={() => setActiveTab('config')}
            data-tooltip="Configure application settings and manage custom models"
          >
            Config
          </button>
        </div>

      </div>

      <div className='seperator'></div>

        <div className="centerContent">

        {/* Wrap the content in conditional rendering based on active tab */}
        {activeTab === 'about' && (
          <div className="about-container">

            <div className="about-header">
              <div className="about-header-sub">
                <h2>Welcome to Upscalee</h2>
                <img src={"https://cdnl.iconscout.com/lottie/premium/thumb/flying-rocket-7551854-6158886.gif"} alt="Logo" className="logoo" height={'80rem'}></img>
              </div>
              <p>Your AI-powered image upscaling solution</p>
            </div>

            <div className="features-grid">
              <div className="feature-card">
                <h3>
                  <span role="img" aria-label="ai">🤖</span>
                  AI-Powered Upscaling
                </h3>
                <p>Transform your images to higher resolutions using state-of-the-art AI models. Support for multiple architectures including Real-ESRGAN and more.</p>
              </div>
              <div className="feature-card">
                <h3>
                  <span role="img" aria-label="models">🎯</span>
                  Multiple Models
                </h3>
                <p>Choose from various pre-trained models or add your own custom models. Each model is optimized for different types of images and use cases.</p>
              </div>
              <div className="feature-card">
                <h3>
                  <span role="img" aria-label="comparison">⚖️</span>
                  Real-time Comparison
                </h3>
                <p>Compare original and upscaled images side by side with an interactive slider. Instantly see the improvements in image quality.</p>
              </div>
              <div className="feature-card">
                <h3>
                  <span role="img" aria-label="quality">📊</span>
                  Quality Assessment
                </h3>
                <p>Analyze the quality of upscaled images with built-in assessment tools. Make informed decisions about your upscaling results.</p>
              </div>
            </div>

            <div className="getting-started">

              <h2>Getting Started</h2>
              <ol>
                <li>Select an AI model from the available options in the dropdown menu</li>
                <li>Upload your image using the drag-and-drop interface or file picker</li>
                <li>Choose your desired upscaling settings (if available for the selected model)</li>
                <li>Click "Upscale Image" to process your image</li>
                <li>Use the comparison slider to view the results and save your upscaled image</li>
              </ol>
              <button
                className="start-upscaling-btn"
                onClick={() => {
                  document.body.scrollTop = 0;
                  document.documentElement.scrollTop = 0;
                  setActiveTab('upscale');
                }}
              >
                Start Upscaling! 🚀
              </button>
              <img src={cat} alt="cat" className='cat'></img>
            </div>

            <div className="tabs-explanation">
              <h2>Navigation Guide</h2>
              <div className="tabs-grid">
                <div
                  className="tab-explanation-card"
                  onClick={() => {
                    document.body.scrollTop = 0; // For Safari
                    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
                    setActiveTab('upscale');
                  }}
                >
                  <div className="tab-icon">🔍</div>
                  <h3>Upscale</h3>
                  <p>The main workspace where you can enhance your images. Select AI models, upload images, and control upscaling settings for optimal results.</p>
                </div>

                <div
                  className="tab-explanation-card"
                  onClick={() => {
                    document.body.scrollTop = 0;
                    document.documentElement.scrollTop = 0;
                    setActiveTab('model-info');
                  }}
                >
                  <div className="tab-icon">🤖</div>
                  <h3>Models</h3>
                  <p>Browse and select from available AI models. Each model is optimized for different types of images and upscaling scenarios.</p>
                </div>

                <div
                  className="tab-explanation-card"
                  onClick={() => {
                    document.body.scrollTop = 0;
                    document.documentElement.scrollTop = 0;
                    setActiveTab('info');
                  }}
                >
                  <div className="tab-icon">📊</div>
                  <h3>Image Info</h3>
                  <p>Analyze detailed information about your images including dimensions, format, and metadata. Compare original and upscaled image properties.</p>
                </div>

                <div
                  className="tab-explanation-card"
                  onClick={() => {
                    document.body.scrollTop = 0;
                    document.documentElement.scrollTop = 0;
                    setActiveTab('config');
                  }}
                >
                  <div className="tab-icon">⚙️</div>
                  <h3>Config</h3>
                  <p>Customize application settings, manage custom models, and configure advanced options for optimal performance.</p>
                </div>
              </div>
            </div>

            <div className="tech-stack">
              <h2>Technology Stack</h2>

              <div className="tech-grid">
                <a href="https://create-react-app.dev/" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={reactLogo} alt="Create React App" className="tech-icon" />
                    <p>Create React App</p>
                  </div>
                </a>
                <a href="https://www.python.org/" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={pythonLogo} alt="Python" className="tech-icon" />
                    <p>Python</p>
                  </div>
                </a>
                <a href="https://pytorch.org/" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={pytorchLogo} alt="PyTorch" className="tech-icon" />
                    <p>PyTorch</p>
                  </div>
                </a>
                <a href="https://flask.palletsprojects.com/en/stable/" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={flaskLogo} alt="Flask" className="tech-icon" />
                    <p>Flask</p>
                  </div>
                </a>
              </div>
              
              <div className="tech-grid">
                <a href="https://github.com/xinntao/Real-ESRGAN" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={"https://github.com/xinntao/Real-ESRGAN/raw/master/assets/realesrgan_logo.png"} alt="Real-ESRGAN" className="tech-icon imgBigger" />
                    <p>Real-ESRGAN</p>
                  </div>
                </a>
                <a href="https://openmodeldb.info/" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={"https://avatars.githubusercontent.com/u/123817276"} alt="OpenModelDB" className="tech-icon" />
                    <p>OpenModelDB</p>
                  </div>
                </a>
                <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={"https://cdn.iconscout.com/icon/premium/png-256-thumb/cuda-11796839-9633028.png?f=webp"} alt="CUDA" className="tech-icon" />
                    <p>CUDA</p>
                  </div>
                </a>
                <a href="https://github.com/chaiNNer-org/spandrel" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={"https://avatars.githubusercontent.com/u/111189700"} alt="Spandrel" className="tech-icon" />
                    <p>Spandrel</p>
                  </div>
                </a>
              </div>

              <div className="tech-grid">
                <a href="https://github.com/chaofengc/IQA-PyTorch" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={IQA_PyTorch} alt="IQA-PyTorch" className="tech-icon imgBigger" />
                    <p>IQA-PyTorch</p>
                  </div>
                </a>
                <a href="https://opencv.org/" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={OpenCV} alt="OpenCV" className="tech-icon" />
                    <p>OpenCV</p>
                  </div>
                </a>
                <a href="" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={ee} alt="EnhanceEverything" className="tech-icon" />
                    <p>EnhanceEverything</p>
                  </div>
                </a>
                <a href="https://catppuccin.com/palette" target="_blank" rel="noopener noreferrer">
                  <div className="tech-item">
                    <img src={catppuccin} alt="Catppuccin" className="tech-icon" />
                    <p>Catppuccin</p>
                  </div>
                </a>
              </div>

            </div>

            <img src={"https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/palette/macchiato.png"}
              width={'100%'}
            >
            </img>

          </div>
        )}

        {activeTab === 'upscale' && (
          <>
            <div className="info-tab">
              <div className="model-selection">
                <div className="model-selection-label">
                  <h2 className='titleSizeColour'>
                    Select Model:
                  </h2>
                </div>
                <div className="model-selection-container">
                  <div
                    className="custom-select"
                    data-tooltip="Hover over model names to see their purpose"
                  >
                    <select
                      id="model-select"
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      disabled={loading}
                    >
                      <option value="" disabled>Select a model...</option>
                      {Object.entries(models).map(([key, info]) => (
                        <option
                          key={key}
                          value={key}
                          title={info.description}
                        >
                          {info.name || key}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="model-description-tooltip" data-tooltip={models[selectedModel]?.description}>
                    ⓘ
                  </div>
                  {showScaleSelector()}
                </div>
              </div>

              {/* Add this new section to display model details
            {selectedModel && models[selectedModel] && (
              <div className="model-info-panel">
                <h3>{models[selectedModel].name}</h3>
                <p className="model-description">{models[selectedModel].description}</p>
                <div className="model-details">
                  <span className="model-scale">
                    Scale: {models[selectedModel].variable_scale ? 
                      "Variable (1-4x)" : 
                      `${models[selectedModel].scale}x`}
                  </span>
                </div>
              </div>
            )} */}

              <div>
                <DropZone
                  onFileSelect={handleFileSelect}
                  previewUrl={imagePreview}
                />

                <div className="button-container">
                  <button
                    onClick={resetInput}
                    disabled={loading}
                    className="reset-button"
                    data-tooltip="Clear the current image and all results to start over"
                  >
                    Reset Input
                  </button>
                  <button
                    onClick={upscaleImage}
                    disabled={!selectedImage || loading}
                    data-tooltip="Process the selected image with the chosen model and scale settings"
                  >
                    Upscale Image
                  </button>
                </div>
              </div>


              {modelDownloading && (
                <div className="model-download-status">
                  <div className="loading-spinner"></div>
                  <p>{downloadProgress}</p>
                </div>
              )}

              <div className="App">
                {imagePreview && upscaledImage ? (
                  <div>
                    <h2 className='image-comparison-header'>Image Comparison</h2>
                    <ImageComparisonSlider
                      leftImage={imagePreview}
                      rightImage={upscaledImage}
                      showQueueButton={true}
                      onQueue={handleQueue}
                      leftLabel="Original"
                      rightLabel={`${selectedModel} (${selectedScale}x)`}
                      modelName={selectedModel}
                      scale={selectedScale}
                      originalFilename={selectedImage?.name}
                    />
                  </div>
                ) : imagePreview && loading ? (
                  <div>
                    <h2 className="image-comparison-header">Image Comparison</h2>
                    <div className="placeholder-container loading-indicators-center-column">

                      <div className="spinner"></div>
                      <br />
                      <p className="loading-text">Upscaling image with {selectedModel} at {selectedScale}x scale...</p>
                    </div>
                  </div>
                ) : imagePreview ? (
                  <div>
                    <h2 className="image-comparison-header">Image Comparison</h2>
                    <div className="placeholder-container">

                      <p>Click "Upscale Image" to start processing</p>
                    </div>
                  </div>
                ) : (
                  <div className="placeholder-message">
                    <p>Select and upscale an image to use the comparison slider</p>
                  </div>
                )}
              </div>

              {showQueue && (
                <ImageQueue
                  queuedImages={queuedImages}
                  onClearQueue={() => {
                    setShowQueue(false);
                    setQueuedImages([]);
                  }}
                />
              )}

            </div>

            {imagePreview && upscaledImage && (
              <ImageQualityAssessment
                originalImage={imagePreview}
                upscaledImage={upscaledImage}
              />
            )}
          </>
        )}

        {activeTab === 'model-info' && (
          <div className="model-info-tab">
            <h2 className='titleSizeColour'>Available Models:</h2>
            <h3>Click on a model card to upscale with it</h3>
            <h3>Click <a href="https://openmodeldb.info/" target="_blank" rel="noopener noreferrer">here</a> to go download some more models</h3>
            <div className="model-cards-container">
              {Object.entries(models).map(([key, model]) => (
                <div
                  key={key}
                  className="model-card"
                  onClick={() => {
                    setSelectedModel(key);
                    setActiveTab('upscale');
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="model-card-header">
                    <h3>{model.name}</h3>
                    <span className="model-type-badge">
                      {model.is_spandrel ? 'Custom Model' : 'Built-in'}
                    </span>
                  </div>

                  <div className="model-card-body">
                    <p className="model-description">{model.description}</p>

                    <div className="model-specs">
                      <div className="spec-item">
                        <span className="spec-label">Scale:</span>
                        <span className="spec-value">
                          {model.variable_scale ? 'Variable (1-4x)' : `${model.scale}x`}
                        </span>
                      </div>

                      {model.architecture && (
                        <div className="spec-item">
                          <span className="spec-label">Architecture:</span>
                          <span className="spec-value">{model.architecture}</span>
                        </div>
                      )}

                      {model.source_url && (
                        <div className="spec-item">
                          <span className="spec-label">Source:</span>
                          <span className="spec-value">
                            <a
                              href={model.source_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()} // Prevent card click when clicking link
                              className="source-link"
                            >
                              Link ↗
                            </a>
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'info' && (

          <div className="info-tab">
            <h2 className="info-tab-header titleSizeColour">Image Information</h2>

            <DropZone
              onFileSelect={handleFileSelect}
              previewUrl={imagePreview}
            />

            <div className="button-container">
              <button
                onClick={() => selectedImage && fetchImageInfo(selectedImage)}
                disabled={!selectedImage || loading}
                data-tooltip="Analyze the selected image to view detailed information"
              >
                Get Image Info
              </button>
            </div>

            {imageInfo ? (
              <div className="image-info">
                {/* Basic Info Section */}
                <section className="info-section">
                  <h3>Basic Information</h3>
                  <p><strong>Format:</strong> {imageInfo.basic_info.format}</p>
                  <p><strong>Format Description:</strong> {imageInfo.basic_info.format_description}</p>
                  <p><strong>Mode:</strong> {imageInfo.basic_info.mode}</p>
                  <p><strong>Color Space:</strong> {imageInfo.basic_info.color_space}</p>
                  {imageInfo.basic_info.animation && (
                    <>
                      <p><strong>Animated:</strong> Yes</p>
                      <p><strong>Number of Frames:</strong> {imageInfo.basic_info.n_frames}</p>
                    </>
                  )}
                </section>

                {/* Dimensions Section */}
                <section className="info-section">
                  <h3>Dimensions</h3>
                  <p><strong>Width:</strong> {imageInfo.dimensions.width}px</p>
                  <p><strong>Height:</strong> {imageInfo.dimensions.height}px</p>
                  <p><strong>Megapixels:</strong> {imageInfo.dimensions.megapixels.toFixed(2)} MP</p>
                  <p><strong>Aspect Ratio:</strong> {imageInfo.dimensions.aspect_ratio}</p>
                  <p><strong>Orientation:</strong> {imageInfo.dimensions.orientation}</p>
                  <p><strong>Resolution Category:</strong> {imageInfo.dimensions.resolution_category}</p>
                </section>

                {/* Color Information */}
                <section className="info-section">
                  <h3>Color Information</h3>
                  <p><strong>Color Depth:</strong> {imageInfo.color_info.color_depth} bits</p>
                  <p><strong>Bits per Pixel:</strong> {imageInfo.color_info.bits_per_pixel}</p>
                  <p><strong>Channels:</strong> {imageInfo.color_info.channels}</p>
                  <p><strong>Channel Names:</strong> {imageInfo.color_info.channel_names.join(', ')}</p>
                  <p><strong>Has Alpha Channel:</strong> {imageInfo.color_info.has_alpha ? 'Yes' : 'No'}</p>
                  <p><strong>Transparency:</strong> {imageInfo.color_info.transparency ? 'Yes' : 'No'}</p>
                  {imageInfo.color_info.palette_size && (
                    <p><strong>Palette Size:</strong> {imageInfo.color_info.palette_size} colors</p>
                  )}
                </section>

                {/* File Information */}
                <section className="info-section">
                  <h3>File Information</h3>
                  <p><strong>Estimated Memory:</strong> {imageInfo.file_info.estimated_memory_mb.toFixed(2)} MB</p>
                  <p><strong>Compression:</strong> {imageInfo.file_info.compression}</p>
                  <p><strong>Compression Details:</strong> {imageInfo.file_info.compression_details}</p>
                  <p><strong>DPI:</strong> {imageInfo.file_info.dpi}</p>

                  {/* Format-specific details */}
                  {Object.entries(imageInfo.file_info.format_details).map(([key, value]) => (
                    <p key={key}><strong>{key.charAt(0).toUpperCase() + key.slice(1)}:</strong> {String(value)}</p>
                  ))}
                </section>

                {/* Statistics Section */}
                {imageInfo.statistics && Object.keys(imageInfo.statistics).length > 0 && (
                  <section className="info-section">
                    <h3>Channel Statistics</h3>
                    {Object.entries(imageInfo.statistics).map(([channel, stats]: [string, any]) => (
                      <div key={channel} className="channel-stats">
                        <h4>{channel.charAt(0).toUpperCase() + channel.slice(1).replace('_', ' ')}</h4>
                        <p><strong>Min Value:</strong> {stats.min}</p>
                        <p><strong>Max Value:</strong> {stats.max}</p>
                        <p><strong>Mean Value:</strong> {stats.mean.toFixed(2)}</p>
                      </div>
                    ))}
                  </section>
                )}

                {/* EXIF Data */}
                {Object.keys(imageInfo.exif).length > 0 && (
                  <section className="info-section">
                    <h3>EXIF Data</h3>
                    {Object.entries(imageInfo.exif).map(([category, data]) => (
                      <div key={category} className="exif-category">
                        <h4>{category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
                        {Object.entries(data as Record<string, string>).map(([key, value]) => (
                          <p key={key}><strong>{key}:</strong> {value}</p>
                        ))}
                      </div>
                    ))}
                  </section>
                )}
              </div>
            ) : (
              <div className="placeholder-message">
                <p>Drop an image or click to select one, then click "Get Image Info" to view its information.</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'config' && (
          <div className="config-tab">
            <h2 className='titleSizeColour'>Configuration</h2>

            {/* Unregistered Models Section */}
            {unregisteredModels.length > 0 && (
              <div className="config-section">
                <h3>Unregistered Models</h3>
                <p
                  className="section-description"
                // data-tooltip="Models found in your model directory that haven't been configured for use. Registration allows you to set custom names, descriptions, and other metadata"
                >
                  The following models were found in your model directory but are not yet registered.
                  Register them to make them available for use.
                </p>
                <div className="unregistered-models-list">
                  {unregisteredModels.map(model => (
                    <div
                      key={model.name}
                      className="unregistered-model-card"
                    // data-tooltip="Click 'Register Model' to configure this model's settings and make it available for use"
                    >
                      <div className="model-info">
                        <h4>{model.name}</h4>
                        <p
                        // data-tooltip="The upscaling factor this model was trained for"
                        >
                          Scale: {model.scale || 'Unknown'}
                        </p>
                        <p
                          className="file-path"
                        // data-tooltip="Location of the model weights file in your model directory"
                        >
                          Path: {model.path}
                        </p>
                      </div>
                      <button
                        className="register-button"
                        onClick={() => {
                          setSelectedUnregisteredModel(model);
                          setShowRegisterModal(true);
                        }}
                        data-tooltip="Open the registration form to configure this model's settings"
                      >
                        Register Model
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Registration Modal */}
            {showRegisterModal && selectedUnregisteredModel && (
              <ModelRegistrationForm
                model={selectedUnregisteredModel}
                onClose={() => {
                  setShowRegisterModal(false);
                  setSelectedUnregisteredModel(null);
                }}
                onRegister={() => {
                  // Now fetchModels is accessible here
                  fetchModels();
                }}
              />
            )}

            {/* Configuration Options */}
            <div className="config-section">
              <h3>General Settings</h3>
              <div className="config-options">
                <div className="config-option">
                  <label
                    className='config-option-label'
                    data-tooltip="Sets the maximum allowed dimension (width or height) for input images. Larger images will be automatically resized"
                  >
                    Maximum Image Dimension:
                  </label>
                  <input
                    type="number"
                    className='fill_w'
                    value={config.maxImageDimension}
                    onChange={(e) => {
                      setConfig(prev => ({
                        ...prev,
                        maxImageDimension: parseInt(e.target.value)
                      }));
                      setIsConfigChanged(true);
                    }}
                    min={256}
                    max={4096}
                    data-tooltip="Enter a value between 256 and 4096 pixels. Higher values require more memory"
                  />
                </div>
                <div className="config-option lp">
                  <label
                    className='config-option-label'
                    data-tooltip="Directory where custom model weights are stored. Models placed here will be detected automatically"
                  >
                    Model Path:
                  </label>
                  <div className="directory-picker">
                    <input
                      type="text"
                      className='fill_w fix_h'
                      value={config.modelPath}
                      onChange={(e) => {
                        setConfig(prev => ({
                          ...prev,
                          modelPath: e.target.value
                        }));
                        setIsConfigChanged(true);
                      }}
                      data-tooltip="Current path to model weights directory"
                      readOnly
                    />
                    <button
                      onClick={async () => {
                        try {
                          const dirHandle = await (window as any).showDirectoryPicker();
                          const dirPath = dirHandle.name;
                          setConfig(prev => ({
                            ...prev,
                            modelPath: dirPath
                          }));
                          setIsConfigChanged(true);
                        } catch (err) {
                          console.error('Error selecting directory:', err);
                        }
                      }}
                      className="directory-picker-button"
                      type="button"
                      data-tooltip="Select a folder containing your custom model weights"
                    >
                      Browse
                    </button>
                  </div>
                </div>
                <button
                  className="save-config-button"
                  onClick={handleSaveConfig}
                  disabled={!isConfigChanged}
                  data-tooltip="Save your configuration changes. Button is disabled when no changes have been made"
                >
                  Save Configuration
                </button>
              </div>
            </div>

          </div>
        )}

      </div>
    </div>
  );
};

export default ImageUpscaler;
