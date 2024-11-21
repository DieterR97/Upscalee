import React, { useRef, useState, useEffect, useCallback } from 'react';
import './ImageComparisonSlider.css';
import { toast, ToastContainer } from 'react-toastify';

// Define the props interface for the component
interface ImageComparisonSliderProps {
    leftImage: string;
    rightImage: string;
    onQueue?: () => void;
    showQueueButton?: boolean;
    leftLabel?: string;
    rightLabel?: string;
    modelName?: string;
    scale: number;
    originalFilename?: string;
    onCompareClick?: () => void;
    mode?: 'slider' | 'switch' | 'diff';
    onModeChange?: (mode: 'slider' | 'switch' | 'diff') => void;
}

const ImageComparisonSlider: React.FC<ImageComparisonSliderProps> = ({ leftImage, rightImage, onQueue, showQueueButton = false, leftLabel, rightLabel, modelName = 'unknown', scale: initialScale = 0, originalFilename, onCompareClick, mode, onModeChange }) => {
    // Refs for DOM elements
    const containerRef = useRef<HTMLDivElement>(null);
    const sliderRef = useRef<HTMLInputElement>(null);
    const imageWrapperRef = useRef<HTMLDivElement>(null);
    const rightImageRef = useRef<HTMLImageElement>(null);
    const sliderHandleRef = useRef<HTMLDivElement>(null);

    // State variables for component functionality
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [scale, setScale] = useState(1);
    const [isDragging, setIsDragging] = useState(false);
    const [isPanning, setIsPanning] = useState(false);
    const [panX, setPanX] = useState(0);
    const [panY, setPanY] = useState(0);

    const [startX, setStartX] = useState(0);
    const [startY, setStartY] = useState(0);
    const [startLayerX, setStartLayerX] = useState(0);
    const [sliderPosition, setSliderPosition] = useState(50);

    const [currentMode, setCurrentMode] = useState<'slider' | 'switch' | 'diff'>('slider');
    const [currentImage, setCurrentImage] = useState<'left' | 'right'>('left');
    const [diffCanvas, setDiffCanvas] = useState<HTMLCanvasElement | null>(null);

    // Effect to handle fullscreen changes
    useEffect(() => {
        const handleFullscreenChange = () => {
            if (!document.fullscreenElement) {
                setIsFullscreen(false);
                updateTransform();
            }
        };

        document.addEventListener('fullscreenchange', handleFullscreenChange);
        return () => {
            document.removeEventListener('fullscreenchange', handleFullscreenChange);
        };
    }, []);

    // Function to update slider position and trigger clip path update
    const updateSliderPosition = useCallback((value: number) => {
        setSliderPosition(value);
        updateClipPath();
    }, []);

    // Function to update the clip path of the right image based on slider position
    const updateClipPath = useCallback(() => {
        if (rightImageRef.current && containerRef.current && imageWrapperRef.current) {
            // Don't apply clip path in switch or diff mode
            if (currentMode === 'switch' || currentMode === 'diff') {
                rightImageRef.current.style.clipPath = 'none';
                return;
            }

            const containerRect = containerRef.current.getBoundingClientRect();
            const imageRect = imageWrapperRef.current.getBoundingClientRect();

            // Calculate relative slider position accounting for image pan and zoom
            const sliderPositionX = (sliderPosition / 100) * containerRect.width;
            const imageOffsetX = (imageRect.width - containerRect.width) / 2 - panX;
            const relativeSliderPosition = (sliderPositionX + imageOffsetX) / imageRect.width;

            // Clip the right image to create the sliding effect
            const clipValue = Math.max(0, Math.min(1, relativeSliderPosition)) * 100;
            rightImageRef.current.style.transition = 'clip-path 0.1s ease-out';
            rightImageRef.current.style.clipPath = `inset(0 0 0 ${clipValue}%)`;
        }
    }, [sliderPosition, panX, scale, currentMode]);

    // Schedule clip path update on next animation frame
    const scheduleUpdate = useCallback(() => {
        requestAnimationFrame(updateClipPath);
    }, [updateClipPath]);

    // Update transform of image wrapper (for panning and zooming)
    const updateTransform = useCallback(() => {
        if (containerRef.current && imageWrapperRef.current) {
            const containerRect = containerRef.current.getBoundingClientRect();
            const imageRect = imageWrapperRef.current.getBoundingClientRect();

            const maxPanX = Math.max(0, (imageRect.width - containerRect.width) / 2);
            const maxPanY = Math.max(0, (imageRect.height - containerRect.height) / 2);

            setPanX(prev => Math.max(-maxPanX, Math.min(maxPanX, prev)));
            setPanY(prev => Math.max(-maxPanY, Math.min(maxPanY, prev)));

            imageWrapperRef.current.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        }
        scheduleUpdate();
    }, [panX, panY, scale, scheduleUpdate]);

    // Handle slider input changes
    const handleSliderInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const newPosition = Number(e.target.value);
        setSliderPosition(newPosition);
        scheduleUpdate();
    }, [scheduleUpdate]);

    // Start dragging (slider or panning)
    const startDragging = useCallback((e: React.MouseEvent | React.TouchEvent) => {
        e.preventDefault();
        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;

        if (e.target === sliderHandleRef.current) {
            setIsDragging(true);
            setStartX(clientX);
            setStartLayerX(sliderHandleRef.current.offsetLeft);
        } else if (scale > 1) {
            setIsPanning(true);
            setStartX(clientX);
            setStartY('touches' in e ? e.touches[0].clientY : e.clientY);
        }
    }, [scale]);

    // Stop dragging
    const stopDragging = useCallback(() => {
        setIsDragging(false);
        setIsPanning(false);
    }, []);

    // Handle dragging (slider or panning)
    const drag = useCallback((e: MouseEvent | TouchEvent) => {
        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

        if (isDragging && containerRef.current && sliderRef.current) {
            const containerRect = containerRef.current.getBoundingClientRect();
            const layerX = clientX - containerRect.left;
            const percentage = Math.max(0, Math.min(100, (layerX / containerRect.width) * 100));
            
            sliderRef.current.value = percentage.toString();
            updateSliderPosition(percentage);
            scheduleUpdate();
        } else if (isPanning && scale > 1) {
            setPanX(prev => prev + clientX - startX);
            setPanY(prev => prev + clientY - startY);
            setStartX(clientX);
            setStartY(clientY);
            updateTransform();
        }
    }, [isDragging, isPanning, scale, updateSliderPosition, scheduleUpdate, updateTransform]);

    // Handle panning
    const handlePan = useCallback((e: MouseEvent | TouchEvent) => {
        if (isPanning) {
            const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
            const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

            setPanX(prev => prev + clientX - startX);
            setPanY(prev => prev + clientY - startY);
            setStartX(clientX);
            setStartY(clientY);

            scheduleUpdate();
        }
    }, [isPanning, startX, startY, scheduleUpdate]);

    // Handle zooming with mouse wheel
    const zoom = useCallback((event: React.WheelEvent) => {
        event.preventDefault();
        const delta = Math.sign(event.deltaY);
        const oldScale = scale;
        // Limit zoom between 1x and 25x
        setScale(prev => Math.max(1, Math.min(prev - delta * 0.1, 25)));

        // Adjust pan position to keep zoom centered on mouse position
        if (containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;

            setPanX(prev => mouseX - (mouseX - prev) * (scale / oldScale));
            setPanY(prev => mouseY - (mouseY - prev) * (scale / oldScale));
        }

        updateTransform();
    }, [scale, updateTransform]);

    // Reset zoom and pan
    const resetZoom = useCallback(() => {
        setScale(1);
        setPanX(0);
        setPanY(0);
        setSliderPosition(50);
        if (imageWrapperRef.current) {
            imageWrapperRef.current.style.transform = `translate(0px, 0px) scale(1)`;
        }
        scheduleUpdate();
    }, [scheduleUpdate]);

    // Toggle fullscreen mode
    const toggleFullscreen = useCallback(() => {
        if (!isFullscreen) {
            if (containerRef.current?.requestFullscreen) {
                containerRef.current.requestFullscreen();
            }
            containerRef.current?.classList.add('fullscreen');
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
            containerRef.current?.classList.remove('fullscreen');
        }
        setIsFullscreen(prev => !prev);
        updateTransform();
    }, [isFullscreen, updateTransform]);

    // Effect to add and remove event listeners
    useEffect(() => {
        document.addEventListener('mousemove', drag);
        document.addEventListener('touchmove', drag, { passive: false });
        document.addEventListener('mousemove', handlePan);
        document.addEventListener('touchmove', handlePan);
        document.addEventListener('mouseup', stopDragging);
        document.addEventListener('touchend', stopDragging);

        return () => {
            document.removeEventListener('mousemove', drag);
            document.removeEventListener('touchmove', drag);
            document.removeEventListener('mousemove', handlePan);
            document.removeEventListener('touchmove', handlePan);
            document.removeEventListener('mouseup', stopDragging);
            document.removeEventListener('touchend', stopDragging);
        };
    }, [handlePan, stopDragging, drag]);

    // Effect to handle Escape key press in fullscreen mode
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && isFullscreen) {
                toggleFullscreen();
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [isFullscreen, toggleFullscreen]);

    // Effect to initialize slider position
    useEffect(() => {
        updateSliderPosition(50); // Initialize slider position
    }, [updateSliderPosition]);

    // Effect to update clip path when zooming, panning, or changing slider position
    useEffect(() => {
        scheduleUpdate();
    }, [scale, panX, panY, sliderPosition, scheduleUpdate]);

    useEffect(() => {
        const container = containerRef.current;
        if (container) {
            const preventScroll = (e: WheelEvent) => {
                e.preventDefault();
            };

            container.addEventListener('wheel', preventScroll, { passive: false });

            return () => {
                container.removeEventListener('wheel', preventScroll);
            };
        }
    }, []);

    // Add this function inside the ImageComparisonSlider component
    const handleDownload = () => {
        console.log('originalFilename:', originalFilename);
        const baseFilename = originalFilename?.replace('.jpg', '') || 'image';
        console.log('baseFilename:', baseFilename);
        const filename = `${baseFilename}_${modelName}_x${initialScale}.png`;
        console.log('final filename:', filename);
        
        const link = document.createElement('a');
        link.href = rightImage;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    // Modify the Queue button click handler
    const handleCompareClick = () => {
        if (onQueue) {
            onQueue();
        }
        
        // Add scrolling behavior
        setTimeout(() => {
            const imageQueue = document.querySelector('.image-queue');
            if (imageQueue) {
                imageQueue.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
            }
        }, 100);
    };

    // Add this new function to handle snapshot creation
    const handleSnapshot = () => {
        if (!containerRef.current || !rightImageRef.current) return;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Get the natural dimensions of the images
        const leftImg = containerRef.current.querySelector('.image-left') as HTMLImageElement;
        const rightImg = rightImageRef.current;
        
        const width = Math.max(leftImg.naturalWidth, rightImg.naturalWidth);
        const height = Math.max(leftImg.naturalHeight, rightImg.naturalHeight);
        
        // Set canvas dimensions to match image size
        canvas.width = width;
        canvas.height = height;

        // Draw background
        ctx.fillStyle = 'var(--ctp-mocha-crust)';
        ctx.fillRect(0, 0, width, height);

        // Calculate the split position in actual pixels
        const splitX = Math.floor(width * (sliderPosition / 100));
        
        // Draw left image
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, splitX, height);
        ctx.clip();
        ctx.drawImage(leftImg, 0, 0, width, height);
        ctx.restore();

        // Draw right image
        ctx.save();
        ctx.beginPath();
        ctx.rect(splitX, 0, width - splitX, height);
        ctx.clip();
        ctx.drawImage(rightImg, 0, 0, width, height);
        ctx.restore();

        // Draw slider line
        ctx.fillStyle = 'var(--ctp-mocha-blue)';
        ctx.fillRect(splitX - 2, 0, 4, height);

        // Create filename using model info if available
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `comparison-${modelName}-${scale}x-${timestamp}.png`;

        // Download the snapshot
        const link = document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        link.click();

        // Show success toast
        toast.success('Snapshot saved successfully!', {
            position: "top-center",
            autoClose: 3000,
        });
    };

    const calculateDifference = useCallback(() => {
        if (!leftImage || !rightImage) return;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const img1 = new Image();
        const img2 = new Image();

        img1.onload = () => {
            img2.onload = () => {
                // Set canvas size to match images
                canvas.width = img1.width;
                canvas.height = img1.height;

                // Draw and get data from first image
                ctx.drawImage(img1, 0, 0);
                const imageData1 = ctx.getImageData(0, 0, canvas.width, canvas.height);

                // Clear canvas and draw second image
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img2, 0, 0);
                const imageData2 = ctx.getImageData(0, 0, canvas.width, canvas.height);

                // Calculate differences
                const diff = ctx.createImageData(canvas.width, canvas.height);
                for (let i = 0; i < imageData1.data.length; i += 4) {
                    // Calculate color differences
                    const diffR = Math.abs(imageData1.data[i] - imageData2.data[i]);
                    const diffG = Math.abs(imageData1.data[i + 1] - imageData2.data[i + 1]);
                    const diffB = Math.abs(imageData1.data[i + 2] - imageData2.data[i + 2]);

                    // Calculate average difference
                    const avgDiff = (diffR + diffG + diffB) / 3;

                    // Apply threshold for better visibility
                    const threshold = 10; // Adjust this value to control sensitivity
                    const intensity = avgDiff > threshold ? 255 : 0;

                    // Set difference pixel colors (using red for visibility)
                    diff.data[i] = intensity;     // Red
                    diff.data[i + 1] = 0;         // Green
                    diff.data[i + 2] = 0;         // Blue
                    diff.data[i + 3] = intensity > 0 ? 128 : 0;  // Alpha (semi-transparent)
                }

                // Put the difference data back on the canvas
                ctx.putImageData(diff, 0, 0);
                setDiffCanvas(canvas);
            };
            img2.src = rightImage;
        };
        img1.src = leftImage;
    }, [leftImage, rightImage]);

    const handleModeChange = (newMode: 'slider' | 'switch' | 'diff') => {
        setCurrentMode(newMode);
        
        // Remove clip path when switching to switch or diff mode
        if (newMode === 'switch' || newMode === 'diff') {
            if (rightImageRef.current) {
                rightImageRef.current.style.clipPath = 'none';
                rightImageRef.current.style.transition = 'none';
            }
        } else {
            updateSliderPosition(50);
        }

        if (newMode === 'switch') {
            setCurrentImage('left');
        } else if (newMode === 'diff') {
            calculateDifference();
        }
    };

    // Add useEffect to recalculate difference when images change
    useEffect(() => {
        if (currentMode === 'diff') {
            calculateDifference();
        }
    }, [leftImage, rightImage, currentMode, calculateDifference]);

    // Modify the keyboard event listener effect
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Only handle arrow keys if we're in switch mode
            // AND the active element is not a form control
            if (currentMode === 'switch' && 
                !(e.target instanceof HTMLSelectElement) && 
                !(e.target instanceof HTMLInputElement)) {
                
                // Prevent default behavior for arrow keys
                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                    e.preventDefault();
                    
                    if (e.key === 'ArrowLeft') {
                        setCurrentImage('left');
                    } else if (e.key === 'ArrowRight') {
                        setCurrentImage('right');
                    }
                }
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [currentMode]);

    // Add an effect to handle mode changes
    useEffect(() => {
        if (currentMode === 'switch' && rightImageRef.current) {
            rightImageRef.current.style.clipPath = 'none';
            rightImageRef.current.style.transition = 'none';
        }
    }, [currentMode]);

    // Render component
    return (
        <>
            <div className="mode-selection">
                <button 
                    className={`mode-button ${currentMode === 'slider' ? 'active' : ''}`}
                    onClick={() => handleModeChange('slider')}
                    data-tooltip="Compare images using an interactive slider"
                >
                    Slider Mode
                </button>
                <button 
                    className={`mode-button ${currentMode === 'switch' ? 'active' : ''}`}
                    onClick={() => handleModeChange('switch')}
                    data-tooltip="Toggle between original and upscaled images using arrow keys"
                >
                    Switch Mode
                </button>
                <button 
                    className={`mode-button ${currentMode === 'diff' ? 'active' : ''}`}
                    onClick={() => handleModeChange('diff')}
                    data-tooltip="Highlight differences between images. Red areas indicate where changes occurred during upscaling"
                >
                    Difference Mode
                </button>
                {currentMode === 'switch' && (
                    <span className="mode-hint">Use left/right arrow keys to switch</span>
                )}
            </div>
            <div
                ref={containerRef}
                className={`image-comparison-container ${isFullscreen ? 'fullscreen' : ''}`}
                onMouseDown={startDragging}
                onTouchStart={startDragging}
                onWheel={zoom}
                onMouseLeave={stopDragging}
            >
                {/* Add labels */}
                {leftLabel && <div className="image-label left-label">{leftLabel}</div>}
                {rightLabel && <div className="image-label right-label">{rightLabel}</div>}
                
                {/* Image wrapper for panning and zooming */}
                <div 
                    ref={imageWrapperRef} 
                    className="image-wrapper" 
                    style={{transform: `translate(${panX}px, ${panY}px) scale(${scale})`}}
                >
                    <img 
                        src={leftImage} 
                        alt="Image 1" 
                        className="image image-left" 
                        style={{
                            display: currentMode === 'switch' ? 
                                (currentImage === 'left' ? 'block' : 'none') : 
                                'block'
                        }}
                    />
                    <img 
                        ref={rightImageRef} 
                        src={rightImage} 
                        alt="Image 2" 
                        className="image image-right"
                        style={{
                            display: currentMode === 'switch' ? 
                                (currentImage === 'right' ? 'block' : 'none') : 
                                'block'
                        }}
                    />
                    {/* Move the difference overlay inside the image-wrapper */}
                    {currentMode === 'diff' && diffCanvas && (
                        <img 
                            src={diffCanvas.toDataURL()}
                            alt="Difference overlay"
                            className="difference-overlay active"
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                objectFit: 'contain'
                            }}
                        />
                    )}
                </div>
                
                {/* Only show slider elements in slider mode */}
                {currentMode === 'slider' && (
                    <>
                        <div ref={sliderHandleRef} className="slider-handle" style={{left: `${sliderPosition}%`}}></div>
                        <input
                            ref={sliderRef}
                            type="range"
                            min="0"
                            max="100"
                            value={sliderPosition}
                            className="slider"
                            onChange={handleSliderInput}
                        />
                    </>
                )}
            </div>
            {/* Control buttons */}
            <div className="button-container">
                <button 
                    onClick={toggleFullscreen}
                    data-tooltip="Enter fullscreen mode for a larger view. Press ESC or click again to exit"
                >
                    FULLSCREEN <span className="smaller-text">(Exit with ESC)</span>
                </button>
                <button 
                    onClick={resetZoom}
                    data-tooltip="Reset all image adjustments: zoom level, pan position, and slider position"
                >
                    RESET <span className="smaller-text">(Pan, Zoom and Slider)</span>
                </button>
                <button 
                    onClick={handleSnapshot}
                    data-tooltip="Take a snapshot of the current comparison view"
                >
                    SNAPSHOT
                </button>
                <button 
                    onClick={handleDownload}
                    data-tooltip="Download the upscaled image to your device"
                >
                    DOWNLOAD
                </button>
                {showQueueButton && onQueue && (
                    <button 
                        onClick={handleCompareClick}
                        className="queue-button"
                        data-tooltip="Save this result to compare with outputs from different AI models"
                    >
                        COMPARE MODELS
                    </button>
                )}
            </div>
        </>
    );
};

export default ImageComparisonSlider;
