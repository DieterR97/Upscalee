import React, { useRef, useState } from 'react';
import './DropZone.css';

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  previewUrl?: string | null;
}

const DropZone: React.FC<DropZoneProps> = ({ onFileSelect, previewUrl }) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      className={`drop-zone ${isDragging ? 'dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept="image/*"
        style={{ display: 'none' }}
      />
      {previewUrl ? (
        <img 
          src={previewUrl} 
          alt="Selected" 
          className="drop-zone-preview"
        />
      ) : (
        <div className="dropzone_text">
          <p>Drop an image here or click to select</p>
          <p className="supported-formats">Supported formats: PNG, JPG, JPEG, WebP</p>
        </div>
      )}
    </div>
  );
};

export default DropZone;
