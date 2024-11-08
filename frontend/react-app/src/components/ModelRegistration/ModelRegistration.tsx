import React, { useState } from 'react';
import { UnregisteredModel } from '../../interfaces';
import { toast } from 'react-toastify';
import './ModelRegistration.css';

interface Props {
  unregisteredModels: UnregisteredModel[];
  onModelRegistered: () => void;
}

export const ModelRegistration: React.FC<Props> = ({ unregisteredModels, onModelRegistered }) => {
  const [registrationData, setRegistrationData] = useState<Record<string, {
    displayName: string;
    description: string;
    scale: number;
    variableScale: boolean;
    type: string;
    architecture: string;
  }>>({});

  const handleRegister = async (fileName: string) => {
    const data = registrationData[fileName];
    if (!data) return;

    try {
      const response = await fetch('http://localhost:5000/register-custom-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...data,
          name: fileName.replace('.pth', ''),
          fileName: fileName,
        }),
      });

      if (response.ok) {
        toast.success('Model registered successfully');
        onModelRegistered();
      } else {
        toast.error('Failed to register model');
      }
    } catch (error) {
      toast.error('Error registering model');
      console.error('Error:', error);
    }
  };

  if (unregisteredModels.length === 0) return null;

  return (
    <div className="model-registration">
      <h3>Unregistered Models Found</h3>
      <div className="unregistered-models-list">
        {unregisteredModels.map((model) => (
          <div key={model.file_name} className="unregistered-model-card">
            <h4>{model.file_name}</h4>
            <div className="registration-form">
              <input
                type="text"
                placeholder="Display Name"
                data-tooltip="A user-friendly name for the model that will be shown in the interface"
                onChange={(e) => setRegistrationData(prev => ({
                  ...prev,
                  [model.file_name]: {
                    ...prev[model.file_name],
                    displayName: e.target.value
                  }
                }))}
              />
              <textarea
                placeholder="Description"
                data-tooltip="Describe what the model does best (e.g., 'Specialized for anime-style images' or 'Best for photo restoration')"
                onChange={(e) => setRegistrationData(prev => ({
                  ...prev,
                  [model.file_name]: {
                    ...prev[model.file_name],
                    description: e.target.value
                  }
                }))}
              />
              <input
                type="text"
                placeholder="Architecture (e.g., ESRGAN, SwinIR, HAT)"
                data-tooltip="The neural network architecture used by this model (e.g., ESRGAN, SwinIR, HAT, SPAN)"
                onChange={(e) => setRegistrationData(prev => ({
                  ...prev,
                  [model.file_name]: {
                    ...prev[model.file_name],
                    architecture: e.target.value
                  }
                }))}
              />
              <input
                type="number"
                placeholder="Scale (e.g., 4)"
                data-tooltip="The upscaling factor this model was trained for (e.g., 2 for 2x, 4 for 4x upscaling)"
                onChange={(e) => setRegistrationData(prev => ({
                  ...prev,
                  [model.file_name]: {
                    ...prev[model.file_name],
                    scale: parseInt(e.target.value)
                  }
                }))}
              />
              <div className="checkbox-container">
                <label data-tooltip="Enable if the model supports variable upscaling factors. Most models are trained for a fixed scale">
                  <input
                    type="checkbox"
                    onChange={(e) => setRegistrationData(prev => ({
                      ...prev,
                      [model.file_name]: {
                        ...prev[model.file_name],
                        variableScale: e.target.checked
                      }
                    }))}
                  />
                  Variable Scale
                </label>
              </div>
              <select
                data-tooltip="The type of images this model works best with"
                onChange={(e) => setRegistrationData(prev => ({
                  ...prev,
                  [model.file_name]: {
                    ...prev[model.file_name],
                    type: e.target.value
                  }
                }))}
              >
                <option value="">Select Type</option>
                <option value="general">General</option>
                <option value="anime">Anime</option>
                <option value="photo">Photo</option>
              </select>
              <button
                onClick={() => handleRegister(model.file_name)}
                disabled={!registrationData[model.file_name] || 
                  !registrationData[model.file_name].displayName ||
                  !registrationData[model.file_name].description ||
                  !registrationData[model.file_name].architecture ||
                  !registrationData[model.file_name].scale ||
                  !registrationData[model.file_name].type}
                data-tooltip="Register this model with the provided information. All fields must be filled out"
              >
                Register Model
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};