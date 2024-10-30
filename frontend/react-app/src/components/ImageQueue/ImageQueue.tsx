import React, { useState } from 'react';
import ImageComparisonSlider from '../ImageComparisonSlider/ImageComparisonSlider';
import './ImageQueue.css';

interface QueuedImageInfo {
    modelName: string;
    scale: number;
    originalImage: string;
    upscaledImage: string;
    originalFilename?: string;
}

interface ImageQueueProps {
    queuedImages: QueuedImageInfo[];
    onClearQueue: () => void;
}

const ImageQueue: React.FC<ImageQueueProps> = ({ queuedImages, onClearQueue }) => {
    const [selectedImages, setSelectedImages] = useState<[number, number] | null>(null);

    const compareImages = (index1: number, index2: number) => {
        setSelectedImages([index1, index2]);
    };

    return (
        <div className="image-queue">
            <h3>Queued Images</h3>
            
            {queuedImages.length > 0 ? (
                <>
                    <div className="queue-list">
                        {queuedImages.map((img, index) => (
                            <div key={index} className="queue-item">
                                <img 
                                    src={img.upscaledImage} 
                                    alt={`${img.modelName} x${img.scale}`}
                                    className="thumbnail"
                                />
                                <div className="queue-item-info">
                                    <p>{img.modelName}</p>
                                    <p>Scale: x{img.scale}</p>
                                </div>
                                <div className="queue-item-actions">
                                    {queuedImages.length > 1 && (
                                        <select 
                                            onChange={(e) => compareImages(index, parseInt(e.target.value))}
                                            value=""
                                        >
                                            <option value="">Compare with...</option>
                                            {queuedImages.map((_, idx) => 
                                                idx !== index && (
                                                    <option key={idx} value={idx}>
                                                        {queuedImages[idx].modelName} x{queuedImages[idx].scale}
                                                    </option>
                                                )
                                            )}
                                        </select>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                    
                    <button onClick={onClearQueue} className="clear-queue-btn">
                        Clear Queue
                    </button>

                    {selectedImages && (
                        <div className="comparison-view">
                            <h4>Comparison View</h4>
                            <ImageComparisonSlider
                                leftImage={queuedImages[selectedImages[0]].upscaledImage}
                                rightImage={queuedImages[selectedImages[1]].upscaledImage}
                                leftLabel={`${queuedImages[selectedImages[0]].modelName} (${queuedImages[selectedImages[0]].scale}x)`}
                                rightLabel={`${queuedImages[selectedImages[1]].modelName} (${queuedImages[selectedImages[1]].scale}x)`}
                                scale={queuedImages[selectedImages[0]].scale}
                                originalFilename={queuedImages[selectedImages[0]].originalFilename}
                                modelName={queuedImages[selectedImages[0]].modelName}
                            />
                        </div>
                    )}
                </>
            ) : (
                <p>No images in queue</p>
            )}
        </div>
    );
};

export default ImageQueue;
