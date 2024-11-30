import React, { useState, useEffect } from 'react';
import './ImageQualityAssessment.css';

interface MetricInfo {
  name: string;
  description: string;
  score_range: [number, number];
  type: 'nr' | 'fr';
  higher_better: boolean;
  category: string;
}

interface MetricScore {
  type: 'fr' | 'nr';
  score?: number;  // for FR metrics
  original?: number;  // for NR metrics
  upscaled?: number;  // for NR metrics
  error?: string;
}

interface Scores {
  [key: string]: MetricScore;
}

interface ImageQualityAssessmentProps {
  originalImage: string | null;
  upscaledImage: string | null;
}

export const ImageQualityAssessment: React.FC<ImageQualityAssessmentProps> = ({
  originalImage,
  upscaledImage,
}) => {
  const [scores, setScores] = useState<Scores>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metricInfo, setMetricInfo] = useState<{ [key: string]: MetricInfo }>({});
  const [selectedMetrics, setSelectedMetrics] = useState<{
    nr: string[];
    fr: string[];
  }>({ nr: [], fr: [] });
  const [metricsToRun, setMetricsToRun] = useState<{
    nr: string[];
    fr: string[];
  }>({ nr: [], fr: [] });
  const [loadingMetrics, setLoadingMetrics] = useState<string[]>([]);

  // Fetch metric information and selected metrics
  useEffect(() => {
    const fetchMetricData = async () => {
      try {
        const [infoResponse, selectedResponse] = await Promise.all([
          fetch('http://localhost:5000/metrics/catalog'),
          fetch('http://localhost:5000/metrics/selected')
        ]);
        
        if (infoResponse.ok && selectedResponse.ok) {
          const [infoData, selectedData] = await Promise.all([
            infoResponse.json(),
            selectedResponse.json()
          ]);
          
          setMetricInfo(infoData);
          setSelectedMetrics(selectedData);
          setMetricsToRun({ nr: [], fr: [] });
        }
      } catch (error) {
        console.error('Error fetching metric data:', error);
        setError('Failed to load metric information');
      }
    };

    fetchMetricData();
  }, []);

  const handleMetricToggle = (metricId: string, type: 'nr' | 'fr') => {
    setMetricsToRun(prev => {
      const newMetrics = { ...prev };
      if (newMetrics[type].includes(metricId)) {
        newMetrics[type] = newMetrics[type].filter(id => id !== metricId);
      } else {
        newMetrics[type] = [...newMetrics[type], metricId];
      }
      return newMetrics;
    });
  };

  const getBase64FromImage = (imgUrl: string): Promise<string> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }
        ctx.drawImage(img, 0, 0);
        resolve(canvas.toDataURL('image/jpeg'));
      };
      img.onerror = reject;
      img.src = imgUrl;
    });
  };

  const checkMetricStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/metrics/status');
      const data = await response.json();
      if (data.loading && data.loading_metrics?.length > 0) {
        setLoadingMetrics(data.loading_metrics);
        // Poll again in a second
        setTimeout(checkMetricStatus, 1000);
      } else {
        setLoadingMetrics([]);
      }
    } catch (error) {
      console.error('Error checking metric status:', error);
    }
  };

  const calculateMetrics = async () => {
    if (!originalImage || !upscaledImage) {
      setError('Both original and upscaled images are required');
      return;
    }

    // Scroll to loading indicator
    setTimeout(() => {
      // Scroll to loading indicator immediately after button click
      const loadingElement = document.querySelector('.iqa-loading-status');
      if (loadingElement) {
        loadingElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);

    setLoading(true);
    setError(null);

    let retryCount = 0;
    const maxRetries = 3;
    const retryDelay = 1000; // 1 second

    const attemptCalculation = async (): Promise<void> => {
      try {
        // Convert blob URLs to base64
        const [originalBase64, upscaledBase64] = await Promise.all([
          blobUrlToBase64(originalImage),
          blobUrlToBase64(upscaledImage)
        ]);

        const response = await fetch('http://localhost:5000/calculate-metrics', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            original_image: originalBase64,
            upscaled_image: upscaledBase64,
            metrics: metricsToRun
          })
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to calculate metrics');
        }

        const data = await response.json();
        setScores(data);

        // Scroll to results
        setTimeout(() => {
          const resultsElement = document.querySelector('.results');
          if (resultsElement) {
            resultsElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 100);
      } catch (error) {
        if (retryCount < maxRetries) {
          retryCount++;
          console.log(`Retry attempt ${retryCount}/${maxRetries}...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          return attemptCalculation();
        }
        
        console.error('Error:', error);
        setError(
          'Unable to connect to the server. Please check if the backend service is running and try again in a few moments.'
        );
      }
    };

    try {
      await attemptCalculation();
    } finally {
      setLoading(false);
    }
  };

  // Add this helper function to convert blob URLs to base64
  const blobUrlToBase64 = async (blobUrl: string): Promise<string> => {
    try {
      const response = await fetch(blobUrl);
      const blob = await response.blob();
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = reader.result as string;
          resolve(base64String);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error('Error converting blob to base64:', error);
      throw error;
    }
  };

  const formatScore = (score: number, range: [number, number]) => {
    if (range[1] === 100 && range[0] === 0) {
      return score.toFixed(1);
    }
    return score.toFixed(3);
  };

  const getScoreClass = (score: number, info: MetricInfo) => {
    const midpoint = (info.score_range[1] - info.score_range[0]) / 2;
    return info.higher_better ? 
      (score > midpoint ? 'good' : 'bad') : 
      (score < midpoint ? 'good' : 'bad');
  };

  return (
    <div className="iqa-container">
      {!originalImage || !upscaledImage ? (
        <p className="iqa-message">Upload and upscale an image to perform quality assessment</p>
      ) : (
        <>
          <div className="metric-selection">
            <h3>Select Metrics to Calculate</h3>
            <div className="metric-groups">
              <div className="metric-group">
                <h4>No-Reference (NR) Metrics</h4>
                <p className="metric-group-description">
                  These evaluate single images without needing a reference
                </p>
                {selectedMetrics.nr.map(metricId => {
                  const info = metricInfo[metricId];
                  return (
                    <div key={metricId} className="metric-item">
                      <label>
                        <input
                          type="checkbox"
                          checked={metricsToRun.nr.includes(metricId)}
                          onChange={() => handleMetricToggle(metricId, 'nr')}
                          disabled={loading}
                        />
                        <div className="metric-info">
                          <strong>{info?.name} <span className="metric-id">({metricId})</span></strong>
                          <p>{info?.description}</p>
                          <p className="score-range">
                            Score range: {info?.score_range.join(' to ')}
                            <br />
                            ({info?.higher_better ? 'higher is better' : 'lower is better'})
                          </p>
                        </div>
                      </label>
                    </div>
                  );
                })}
              </div>

              <div className="metric-group">
                <h4>Full-Reference (FR) Metrics</h4>
                <p className="metric-group-description">
                  These compare against a reference image
                </p>
                {selectedMetrics.fr.map(metricId => {
                  const info = metricInfo[metricId];
                  return (
                    <div key={metricId} className="metric-item">
                      <label>
                        <input
                          type="checkbox"
                          checked={metricsToRun.fr.includes(metricId)}
                          onChange={() => handleMetricToggle(metricId, 'fr')}
                          disabled={loading}
                        />
                        <div className="metric-info">
                          <strong>{info?.name} <span className="metric-id">({metricId})</span></strong>
                          <p>{info?.description}</p>
                          <p className="score-range">
                            Score range: {info?.score_range.join(' to ')}
                            <br />
                            ({info?.higher_better ? 'higher is better' : 'lower is better'})
                          </p>
                        </div>
                      </label>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="metrics-actions">
              <button
                className="calculate-metrics-button"
                onClick={calculateMetrics}
                disabled={loading || (metricsToRun.nr.length === 0 && metricsToRun.fr.length === 0)}
              >
                {loading ? 'Processing...' : 'Calculate Selected Metrics'}
              </button>
            </div>
          </div>

          {loading && (
            <div className="iqa-loading-status">
              <div className="loading-spinner"></div>
              <p>
                {loadingMetrics.length > 0 
                  ? `Downloading metric models: ${loadingMetrics.join(', ')}` 
                  : 'Calculating image quality metrics...'}
              </p>
            </div>
          )}

          {error && <div className="error">{error}</div>}

          {Object.keys(scores).length > 0 && (
            <div className="results">
              <h3>Results:</h3>
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Type</th>
                    <th>Score</th>
                    <th>Range</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {[...metricsToRun.nr, ...metricsToRun.fr].map(metricId => {
                    const info = metricInfo[metricId];
                    const result = scores[metricId];
                    
                    if (!info || !result) return null;

                    return (
                      <tr key={metricId}>
                        <td>
                          {info.name} <span className="metric-id">({metricId})</span>
                        </td>
                        <td>{result.type === 'fr' ? 'Full-Reference' : 'No-Reference'}</td>
                        <td>
                          {result.error ? (
                            <span className="error">{result.error}</span>
                          ) : result.type === 'fr' ? (
                            <span className={getScoreClass(result.score!, info)}>
                              {formatScore(result.score!, info.score_range)}
                            </span>
                          ) : (
                            <div className="nr-scores">
                              <div className={getScoreClass(result.original!, info)}>
                                Original: {formatScore(result.original!, info.score_range)}
                              </div>
                              <div className={getScoreClass(result.upscaled!, info)}>
                                Upscaled: {formatScore(result.upscaled!, info.score_range)}
                              </div>
                            </div>
                          )}
                        </td>
                        <td>
                          {info.score_range.join(' - ')}
                          <br />
                          <span className="score-hint">
                            ({info.higher_better ? 'higher is better' : 'lower is better'})
                          </span>
                        </td>
                        <td>{info.description}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ImageQualityAssessment;
