import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

import { createRoot } from 'react-dom/client';

// const root = ReactDOM.createRoot(
//   document.getElementById('root') as HTMLElement
// );
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );

const root = createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <div className="RootParent">
    <React.StrictMode>
      <App />
    </React.StrictMode>
  </div>
);