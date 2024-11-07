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
                onChange={(e) => setRegistrationData(prev => ({
                  ...prev,
                  [model.file_name]: {
                    ...prev[model.file_name],
                    scale: parseInt(e.target.value)
                  }
                }))}
              />
              <label>
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
              <select
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