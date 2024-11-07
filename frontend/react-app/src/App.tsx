import React, { useState, useRef, useEffect } from 'react';
import './style.css';
import ImageComparisonSlider from './components/ImageComparisonSlider/ImageComparisonSlider';
import ImageQualityAssessment from './components/ImageQualityAssessment/ImageQualityAssessment';
import ImageQueue from './components/ImageQueue/ImageQueue';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import DropZone from './components/DropZone/DropZone';

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
  const [activeTab, setActiveTab] = useState<'upscale' | 'model-info' | 'info' | 'config'>('upscale');
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

        // Add toast notification for successful upscale
        toast.success(`Image successfully upscaled using ${selectedModel} (${selectedScale}x)`, {
          position: "top-center",
          autoClose: 3000,
          hideProgressBar: false,
          closeOnClick: true,
          pauseOnHover: true,
          draggable: true,
          progress: undefined,
        });
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
          id="scale-select"
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

    return (
      <div className="modal">
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
                onChange={e => setFormData(prev => ({ ...prev, source_url: e.target.value }))}
                placeholder="e.g., https://github.com/username/repo"
              />
            </div>
            <div className="button-group">
              <button type="submit">Register Model</button>
              <button type="button" onClick={onClose}>Cancel</button>
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

  return (
    <div className="Root">
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
      <div className="centerContent">
        <h1>Image Upscaler</h1>

        {/* Add the tab navigation */}
        <div className="tabs">
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

        {/* Wrap the content in conditional rendering based on active tab */}
        {activeTab === 'upscale' && (
          <>
            <div className="model-selection">
              <div className="model-selection-label">
                <label htmlFor="model-select" className="model-selection-label-text">Select Model: </label>
              </div>
              <div className="model-selection-dropdown">
                <div className="custom-select">
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
                  <div className="model-description-tooltip" data-tooltip={models[selectedModel]?.description}>
                    ⓘ
                  </div>
                </div>
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

            {showScaleSelector()}

            <div>
              <DropZone onFileSelect={handleFileSelect} />

              <div className="button-container">
                <button
                  onClick={resetInput}
                  disabled={loading}
                  className="tooltip-button reset-button"
                  data-tooltip="Clear the current image and all results to start over"
                >
                  Reset Input
                </button>
                <button
                  onClick={upscaleImage}
                  disabled={!selectedImage || loading}
                  className="tooltip-button"
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
              <h1>Image Comparison</h1>
              {imagePreview && upscaledImage ? (
                <>
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
                </>
              ) : imagePreview && loading ? (
                <div className="placeholder-container loading-indicators-center-column">
                  <div className="spinner"></div>
                  <br />
                  <p className="loading-text">Upscaling image with {selectedModel} at {selectedScale}x scale...</p>
                </div>
              ) : imagePreview ? (
                <div className="placeholder-container">
                  <p>Click "Upscale Image" to start processing</p>
                </div>
              ) : (
                <p>Select and upscale an image to use the comparison slider.</p>
              )}
            </div>

            {imagePreview && upscaledImage && (
              <ImageQualityAssessment
                originalImage={imagePreview}
                upscaledImage={upscaledImage}
              />
            )}

            {showQueue && (
              <ImageQueue
                queuedImages={queuedImages}
                onClearQueue={() => {
                  setShowQueue(false);
                  setQueuedImages([]);
                }}
              />
            )}
          </>
        )}

        {activeTab === 'model-info' && (
          <div className="model-info-tab">
            <h2>Available Models:</h2>
            <h3>Click on a model card to upscale with it</h3>
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
            <h2>Image Information</h2>

            <DropZone onFileSelect={handleFileSelect} />

            <div className="button-container">
              <button
                onClick={() => selectedImage && fetchImageInfo(selectedImage)}
                disabled={!selectedImage || loading}
                className="tooltip-button"
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
              <p>Drop an image or click to select, then click "Get Image Info" to view its information.</p>
            )}
          </div>
        )}

        {activeTab === 'config' && (
          <div className="config-tab">
            <h2>Configuration</h2>

            {/* Configuration Options */}
            <div className="config-section">
              <h3>General Settings</h3>
              <div className="config-options">
                <div className="config-option">
                  <label className='config-option-label'>Maximum Image Dimension:</label>
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
                    title="Maximum allowed dimension for input images"
                  />
                </div>
                <div className="config-option">
                  <label className='config-option-label'>Model Path:</label>
                  <input
                    type="text"
                    className='fill_w'
                    value={config.modelPath}
                    onChange={(e) => {
                      setConfig(prev => ({
                        ...prev,
                        modelPath: e.target.value
                      }));
                      setIsConfigChanged(true);
                    }}
                    title="Path to custom model weights"
                  />
                </div>
                <button
                  className="save-config-button"
                  onClick={handleSaveConfig}
                  disabled={!isConfigChanged}
                >
                  Save Configuration
                </button>
              </div>
            </div>

            {/* Unregistered Models Section */}
            {unregisteredModels.length > 0 && (
              <div className="config-section">
                <h3>Unregistered Models</h3>
                <p className="section-description">
                  The following models were found in your model directory but are not yet registered.
                  Register them to make them available for use.
                </p>
                <div className="unregistered-models-list">
                  {unregisteredModels.map(model => (
                    <div key={model.name} className="unregistered-model-card">
                      <div className="model-info">
                        <h4>{model.name}</h4>
                        <p>Scale: {model.scale || 'Unknown'}</p>
                        <p className="file-path">Path: {model.path}</p>
                      </div>
                      <button
                        className="register-button"
                        onClick={() => {
                          setSelectedUnregisteredModel(model);
                          setShowRegisterModal(true);
                        }}
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
          </div>
        )}

      </div>
    </div>
  );
};

export default ImageUpscaler;
