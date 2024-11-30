import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import logo from './assets/upscalee.png'; // Make sure to import your logo

const Root = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [backendStatus, setBackendStatus] = useState<{
    status: string;
    message: string;
    services: Record<string, boolean>;
  } | null>(null);

  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await fetch('http://localhost:5000/health');
        const status = await response.json();
        setBackendStatus(status);
        
        if (status.status === 'ready') {
          // Wait a bit longer to ensure everything is loaded
          setTimeout(() => setIsLoading(false), 500);
        } else {
          // Check again in 1 second
          setTimeout(checkBackendHealth, 1000);
        }
      } catch (error) {
        console.error('Backend health check failed:', error);
        // Retry in 2 seconds if failed
        setTimeout(checkBackendHealth, 2000);
      }
    };

    checkBackendHealth();
  }, []);

  return (
    <div className="RootParent">
      {isLoading && (
        <div className={`loading-overlay ${!isLoading ? 'fade-out' : ''}`}>
          <img src={logo} alt="Upscalee Logo" />
          <div className="loading-spinner" />
          <p>
            {backendStatus
              ? backendStatus.message
              : 'Connecting to backend...'}
          </p>
          {backendStatus?.services && (
            <div className="startup-status">
              {Object.entries(backendStatus.services).map(([service, ready]) => (
                <div key={service} className={`status-item ${ready ? 'ready' : 'pending'}`}>
                  {service.replace('_', ' ')}: {ready ? '✓' : '...'}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <React.StrictMode>
        <App />
      </React.StrictMode>
    </div>
  );
};

const root = createRoot(document.getElementById('root') as HTMLElement);
root.render(<Root />);