import React, { useRef } from 'react';
import './DropZone.css';

interface DropZoneProps {
  onFileSelect: (file: File) => void;
}

const DropZone: React.FC<DropZoneProps> = ({ onFileSelect }) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  return (
    <div
      className="drop-zone"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileSelect}
        accept="image/*"
      />
      <p className="text-gray-500">Drop image here or click to select</p>
    </div>
  );
};

export default DropZone;
