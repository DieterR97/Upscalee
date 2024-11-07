import React, { useRef, useState, useEffect, useCallback } from 'react';
import './ImageComparisonSlider.css';

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
}

const ImageComparisonSlider: React.FC<ImageComparisonSliderProps> = ({ leftImage, rightImage, onQueue, showQueueButton = false, leftLabel, rightLabel, modelName = 'unknown', scale: initialScale = 0, originalFilename }) => {
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
    }, [sliderPosition, panX, scale]);

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

    // Render component
    return (
        <>
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
                <div ref={imageWrapperRef} className="image-wrapper" style={{transform: `translate(${panX}px, ${panY}px) scale(${scale})`}}>
                    <img src={leftImage} alt="Image 1" className="image image-left" />
                    <img ref={rightImageRef} src={rightImage} alt="Image 2" className="image image-right" />
                </div>
                {/* Slider handle */}
                <div ref={sliderHandleRef} className="slider-handle" style={{left: `${sliderPosition}%`}}></div>
                {/* Slider input */}
                <input
                    ref={sliderRef}
                    type="range"
                    min="0"
                    max="100"
                    value={sliderPosition}
                    className="slider"
                    onChange={handleSliderInput}
                />
            </div>
            {/* Control buttons */}
            <div className="button-container">
                <button 
                    onClick={toggleFullscreen}
                    className="tooltip-button"
                    data-tooltip="Enter fullscreen mode for a larger view. Press ESC or click again to exit"
                >
                    FULLSCREEN (Exit with ESC)
                </button>
                <button 
                    onClick={resetZoom}
                    className="tooltip-button"
                    data-tooltip="Reset all image adjustments: zoom level, pan position, and slider position"
                >
                    RESET (Pan, Zoom and Slider)
                </button>
                <button 
                    onClick={handleDownload}
                    className="tooltip-button"
                    data-tooltip="Download the upscaled image to your device"
                >
                    DOWNLOAD
                </button>
                {showQueueButton && onQueue && (
                    <button 
                        onClick={onQueue}
                        className="queue-button tooltip-button"
                        data-tooltip="Add this upscaled image to the queue for comparison with other model outputs"
                    >
                        QUEUE
                    </button>
                )}
            </div>
        </>
    );
};

export default ImageComparisonSlider;
