import React, { useState, useEffect } from 'react';
import './ImageQualityAssessment.css';

interface IQAProps {
    originalImage: string | null;
    upscaledImage: string | null;
}

interface MetricInfo {
    name: string;
    description: string;
    score_range: [number, number];
    type: 'fr' | 'nr';
    higher_better: boolean;
}

interface MetricResult {
    score?: number;  // for FR metrics
    original?: number;  // for NR metrics
    upscaled?: number;  // for NR metrics
    type: 'fr' | 'nr';
    score_range: [number, number];
    error?: string;
}

interface Results {
    [key: string]: MetricResult;
}

interface LoadingState {
    isLoading: boolean;
    message: string;
}

const ImageQualityAssessment: React.FC<IQAProps> = ({ originalImage, upscaledImage }) => {
    // State for storing metrics and results
    const [availableMetrics, setAvailableMetrics] = useState<Record<string, MetricInfo>>({});
    const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
    const [results, setResults] = useState<Results | null>(null);
    const [loading, setLoading] = useState<LoadingState>({ isLoading: false, message: '' });

    // Fetch available metrics on component mount
    useEffect(() => {
        fetch('http://localhost:5000/available-metrics')
            .then(response => response.json())
            .then(data => setAvailableMetrics(data));
    }, []);

    // Toggle metric selection in the checkboxes
    const handleMetricChange = (metric: string) => {
        setSelectedMetrics(prev =>
            prev.includes(metric)
                ? prev.filter(m => m !== metric)
                : [...prev, metric]
        );
    };

    const evaluateQuality = async () => {
        // Validate required inputs
        if (!originalImage || !upscaledImage || selectedMetrics.length === 0) return;

        // Clear previous results before starting new evaluation
        setResults(null);
        setLoading({ isLoading: true, message: 'Evaluating image quality...' });
        
        try {
            const formData = new FormData();
            
            // Convert blob URLs to actual files
            const originalResponse = await fetch(originalImage);
            const upscaledResponse = await fetch(upscaledImage);
            const originalBlob = await originalResponse.blob();
            const upscaledBlob = await upscaledResponse.blob();
            
            formData.append('original_image', originalBlob, 'original.png');
            formData.append('upscaled_image', upscaledBlob, 'upscaled.png');
            formData.append('metrics', JSON.stringify(selectedMetrics));

            // Implement retry logic for handling metric weight downloads
            let isDownloading = true;
            let retryCount = 0;
            const MAX_RETRIES = 30;
            const INITIAL_DELAY = 5000;  // Longer initial delay for first download attempt
            const RETRY_DELAY = 2000;    // Shorter delay for subsequent retries

            while (isDownloading && retryCount < MAX_RETRIES) {
                try {
                    // Add delay based on retry attempt number
                    if (retryCount === 1) {
                        await new Promise(resolve => setTimeout(resolve, INITIAL_DELAY));
                    } else if (retryCount > 1) {
                        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
                    }

                    // Make API request to evaluate image quality
                    const response = await fetch('http://localhost:5000/evaluate-quality', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    console.log(`Attempt ${retryCount + 1} results:`, data);

                    if (data.error) {
                        throw new Error(data.error);
                    }

                    // Check if still downloading weights
                    if (data.downloading?.status) {
                        setLoading({ 
                            isLoading: true, 
                            message: data.downloading.message
                        });
                        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
                        retryCount++;
                    } else {
                        isDownloading = false;
                        setResults(data);
                    }
                } catch (error) {
                    console.error('Error evaluating image quality:', error);
                    setResults(null);
                    setLoading({ 
                        isLoading: false, 
                        message: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`
                    });
                }
            }

            if (retryCount >= MAX_RETRIES) {
                throw new Error('Maximum retries reached while waiting for metric weights download');
            }

        } catch (error) {
            console.error('Error evaluating image quality:', error);
            setResults(null);
            setLoading({ 
                isLoading: false, 
                message: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`
            });
        } finally {
            setLoading(prev => ({ ...prev, isLoading: false }));
        }
    };

    return (
        <div className="iqa-container">
            <h2>Image Quality Assessment</h2>

            <div className="metric-selection">
                <h3>Select Metrics:</h3>
                <div className="metric-groups">
                    <div className="metric-group">
                        <h3>No-Reference (NR) Metrics</h3>
                        <p className="metric-group-description">These evaluate single images without needing a reference</p>
                        {Object.entries(availableMetrics)
                            .filter(([_, info]) => info.type === 'nr')
                            .map(([id, info]) => (
                                <div key={id} className="metric-item">
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={selectedMetrics.includes(id)}
                                            onChange={() => handleMetricChange(id)}
                                            disabled={loading.isLoading}
                                        />
                                        <div className="metric-info">
                                            <strong>{info.name}</strong>
                                            <p>{info.description}</p>
                                            <p className="score-range">
                                                Score range: {info.score_range[0]} to {info.score_range[1]}
                                                ({info.higher_better ? 'higher is better' : 'lower is better'})
                                            </p>
                                        </div>
                                    </label>
                                </div>
                            ))}
                    </div>

                    <div className="metric-group">
                        <h3>Full-Reference (FR) Metrics</h3>
                        <p className="metric-group-description">These compare against a reference image</p>
                        {Object.entries(availableMetrics)
                            .filter(([_, info]) => info.type === 'fr')
                            .map(([id, info]) => (
                                <div key={id} className="metric-item">
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={selectedMetrics.includes(id)}
                                            onChange={() => handleMetricChange(id)}
                                            disabled={loading.isLoading}
                                        />
                                        <div className="metric-info">
                                            <strong>{info.name}</strong>
                                            <p>{info.description}</p>
                                            <p className="score-range">
                                                Score range: {info.score_range[0]} to {info.score_range[1]}
                                                ({info.higher_better ? 'higher is better' : 'lower is better'})
                                            </p>
                                        </div>
                                    </label>
                                </div>
                            ))}
                    </div>
                </div>
            </div>

            <button
                onClick={evaluateQuality}
                disabled={loading.isLoading || selectedMetrics.length === 0}
            >
                {loading.isLoading ? 'Processing...' : 'Evaluate Quality'}
            </button>

            {loading.isLoading && (
                <div className="iqa-loading-status">
                    <div className="loading-spinner"></div>
                    <p>{loading.message}</p>
                </div>
            )}

            {results && (
                <div className="results">
                    <h3>Results:</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Type</th>
                                <th>Score</th>
                                <th>Score Range</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.entries(results)
                                .filter(([key, _]) => key !== 'downloading') // Filter out the downloading status
                                .map(([metric, result]) => {
                                    if (!result || typeof result !== 'object') {
                                        console.log('Skipping invalid result:', metric, result);
                                        return null;
                                    }

                                    const metricInfo = availableMetrics[metric];
                                    if (!metricInfo) {
                                        console.log('Missing metric info for:', metric);
                                        return null;
                                    }

                                    return (
                                        <tr key={metric}>
                                            <td>{metricInfo.name}</td>
                                            <td>{result.type === 'fr' ? 'Full-Reference' : 'No-Reference'}</td>
                                            <td>
                                                {result.error ? (
                                                    <span className="error">{result.error}</span>
                                                ) : result.type === 'fr' ? (
                                                    result.score !== undefined ? result.score.toFixed(4) : 'N/A'
                                                ) : (
                                                    <>
                                                        Original: {result.original !== undefined ? result.original.toFixed(4) : 'N/A'}<br />
                                                        Upscaled: {result.upscaled !== undefined ? result.upscaled.toFixed(4) : 'N/A'}
                                                    </>
                                                )}
                                            </td>
                                            <td>
                                                {`[${metricInfo.score_range[0]}, ${metricInfo.score_range[1]}]`}
                                                <br />
                                                <span className="score-hint">
                                                    ({metricInfo.higher_better ? 'higher is better' : 'lower is better'})
                                                </span>
                                            </td>
                                        </tr>
                                    );
                                })}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default ImageQualityAssessment;
