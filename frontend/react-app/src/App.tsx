import React, { useState, useRef, useEffect } from 'react';
import './style.css';
import ImageComparisonSlider from './components/ImageComparisonSlider/ImageComparisonSlider';
import ImageQualityAssessment from './components/ImageQualityAssessment/ImageQualityAssessment';
import ImageQueue from './components/ImageQueue/ImageQueue';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

interface ModelInfo {
  description: string;
  scale: number;
  variable_scale: boolean;
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

const Tab: React.FC<TabProps> = ({ label, isActive, onClick }) => (
  <button 
    className={`tab ${isActive ? 'active' : ''}`}
    onClick={onClick}
  >
    {label}
  </button>
);

const ImageUpscaler: React.FC = () => {
  // State variables
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [upscaledImage, setUpscaledImage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [models, setModels] = useState<ModelOptions>({});
  const [selectedModel, setSelectedModel] = useState<string>('RealESRGAN_x4plus_anime_6B');
  const [modelDownloading, setModelDownloading] = useState<boolean>(false);
  const [downloadProgress, setDownloadProgress] = useState<string>('');
  const [selectedScale, setSelectedScale] = useState<number>(4);
  const [showQueue, setShowQueue] = useState(false);
  const [queuedImages, setQueuedImages] = useState<QueuedImageInfo[]>([]);
  const [activeTab, setActiveTab] = useState<'upscale' | 'info' | 'config'>('upscale');
  const [imageInfo, setImageInfo] = useState<any>(null);

  // Fetch available models when component mounts
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:5000/models');
        if (response.ok) {
          const modelData = await response.json();
          setModels(modelData);
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };

    fetchModels();
  }, []);

  // Add effect to clear image info and preview when switching tabs
  useEffect(() => {
    if (activeTab === 'info') {
      setImageInfo(null);
      setSelectedImage(null);
      setImagePreview(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // Reset file input
      }
    } else if (activeTab === 'upscale') {
      setShowQueue(false);
      setQueuedImages([]);
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

  // Upscale the selected image
  const upscaleImage = async () => {
    if (!selectedImage) return;

    // Clear the upscaled image before starting the new upscale
    setUpscaledImage(null);

    // Check if model is downloaded
    const isModelDownloaded = await checkModelStatus(selectedModel);
    if (!isModelDownloaded) {
      setModelDownloading(true);
      setDownloadProgress('Downloading model weights... This may take a few minutes.');
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

  // Modify fetchImageInfo to only be called on button click
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
          <Tab 
            label="Upscale" 
            isActive={activeTab === 'upscale'} 
            onClick={() => setActiveTab('upscale')} 
          />
          <Tab 
            label="Image Info" 
            isActive={activeTab === 'info'} 
            onClick={() => setActiveTab('info')} 
          />
          <Tab 
            label="Configuration" 
            isActive={activeTab === 'config'} 
            onClick={() => setActiveTab('config')} 
          />
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
                    {Object.entries(models).map(([key, info]) => (
                      <option 
                        key={key} 
                        value={key}
                        title={info.description}
                      >
                        {key}
                      </option>
                    ))}
                  </select>
                  <div className="model-description-tooltip" data-tooltip={models[selectedModel]?.description}>
                    ⓘ
                  </div>
                </div>
              </div>
            </div>

            {showScaleSelector()}

            <input
              className="file-input"
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              ref={fileInputRef}
              disabled={loading}
            />
            <br />
            <div className="button-container">
              <button 
                onClick={upscaleImage} 
                disabled={!selectedImage || loading}
                className="tooltip-button"
                data-tooltip="Process the selected image with the chosen model and scale settings"
              >
                Upscale Image
              </button>
              <button 
                onClick={resetInput} 
                disabled={loading}
                className="tooltip-button"
                data-tooltip="Clear the current image and all results to start over"
              >
                Reset Input
              </button>
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
                <p>Please select and upscale an image to use the comparison slider.</p>
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

        {activeTab === 'info' && (
          <div className="info-tab">
            <h2>Image Information</h2>
            <div className="button-container">
              <input
                className="file-input"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                ref={fileInputRef}
                disabled={loading}
              />
              <button 
                onClick={() => selectedImage && fetchImageInfo(selectedImage)} 
                disabled={!selectedImage || loading}
              >
                Get Image Info
              </button>
            </div>
            
            {/* Add image preview */}
            {imagePreview && (
              <div className="info-image-preview">
                <img src={imagePreview} alt="Selected image preview" />
              </div>
            )}

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
              <p>Please select an image and click "Get Image Info" to view its information.</p>
            )}
          </div>
        )}

        {activeTab === 'config' && (
          <div className="config-tab">
            <h2>Configuration</h2>
            <div className="config-options">
              <div className="config-option">
                <label>Maximum Image Dimension:</label>
                <input 
                  type="number" 
                  value={1024} 
                  disabled 
                  title="Maximum allowed dimension for input images"
                />
              </div>
              {/* Add more configuration options as needed */}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpscaler;
