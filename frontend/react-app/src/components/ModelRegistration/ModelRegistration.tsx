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
    architecture: string;
    isSpandrelSupported?: boolean;
  }>>({});

  // Initialize form data when unregistered models are loaded
  React.useEffect(() => {
    const initialData: Record<string, any> = {};
    unregisteredModels.forEach(model => {
      // Explicitly log the architecture from spandrel_info
      console.log('Model architecture:', model.spandrel_info?.architecture);
      
      initialData[model.file_name] = {
        displayName: model.name,
        description: '',
        scale: model.spandrel_info?.scale || model.scale || 4,
        variableScale: false,
        // Explicitly set the architecture from spandrel_info
        architecture: model.spandrel_info?.architecture || '',
        isSpandrelSupported: model.spandrel_info?.is_supported
      };
    });
    setRegistrationData(initialData);
  }, [unregisteredModels]);

  return (
    <div className="model-registration">
      <div className="unregistered-models-list">
        {unregisteredModels.map((model) => {
          // Get the current registration data for this model
          const modelData = registrationData[model.file_name] || {};
          // Get the architecture from spandrel_info
          const spandrelArchitecture = model.spandrel_info?.architecture;
          
          return (
            <div key={model.file_name} className="unregistered-model-card">
              <h4>{model.file_name}</h4>
              {model.spandrel_info && (
                <div className="spandrel-info">
                  <span 
                    className="spandrel-badge"
                    data-supported={model.spandrel_info.is_supported}
                  >
                    Spandrel {model.spandrel_info.is_supported ? 'Supported' : 'Unsupported'}
                  </span>
                </div>
              )}
              <div className="registration-form">
                <input
                  type="text"
                  placeholder="Display Name"
                  value={registrationData[model.file_name]?.displayName || ''}
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
                  value={registrationData[model.file_name]?.description || ''}
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
                  // Explicitly use spandrel architecture if available, otherwise use form data
                  value={spandrelArchitecture || modelData.architecture || ''}
                  disabled={!!spandrelArchitecture}
                  data-tooltip={spandrelArchitecture 
                    ? `Architecture detected by Spandrel: ${spandrelArchitecture}` 
                    : "The neural network architecture used by this model"}
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
                  value={registrationData[model.file_name]?.scale || ''}
                  disabled={!!model.spandrel_info?.scale}
                  data-tooltip={model.spandrel_info?.scale 
                    ? `Scale detected by Spandrel: ${model.spandrel_info.scale}` 
                    : "The upscaling factor this model was trained for"}
                  onChange={(e) => setRegistrationData(prev => ({
                    ...prev,
                    [model.file_name]: {
                      ...prev[model.file_name],
                      scale: parseInt(e.target.value)
                    }
                  }))}
                />
                <div className="checkbox-container">
                  <label>
                    <input
                      type="checkbox"
                      checked={registrationData[model.file_name]?.variableScale || false}
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
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};