import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import logo from './assets/upscalee.png'; // Make sure to import your logo

const Root = () => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Wait for the initial render and styles to load
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000); // Adjust timing as needed

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="RootParent">
      {isLoading && (
        <div className={`loading-overlay ${!isLoading ? 'fade-out' : ''}`}>
          <img src={logo} alt="Upscalee Logo" />
          <div className="loading-spinner" />
          <p>Loading Upscalee...</p>
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