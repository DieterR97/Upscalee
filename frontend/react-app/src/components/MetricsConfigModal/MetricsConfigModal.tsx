import React, { useState, useEffect, useMemo } from 'react';
import { Modal } from '../Modal/Modal';
import './MetricsConfigModal.css';

interface MetricInfo {
    name: string;
    description: string;
    score_range: [number, number];
    type: 'nr' | 'fr';
    higher_better: boolean;
    category: string;
}

interface MetricsCatalog {
    [key: string]: MetricInfo;
}

interface SelectedMetrics {
    nr: string[];
    fr: string[];
}

interface MetricsConfigModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const MetricsConfigModal: React.FC<MetricsConfigModalProps> = ({ isOpen, onClose }) => {
    const [catalog, setCatalog] = useState<MetricsCatalog>({});
    const [selected, setSelected] = useState<SelectedMetrics>({ nr: [], fr: [] });
    const [searchTerm, setSearchTerm] = useState('');
    const [categoryFilter, setCategoryFilter] = useState<string>('all');
    const [loading, setLoading] = useState(true);

    // Get unique categories including special filter options
    const categories = useMemo(() => {
        const standardCats = new Set(['all']);
        Object.values(catalog).forEach(metric => standardCats.add(metric.category));
        return [
            'all',
            'No-Reference (NR) Metrics',
            'Full-Reference (FR) Metrics',
            'Selected',
            'Unselected',
            ...Array.from(standardCats).filter(cat => !['all'].includes(cat))
        ];
    }, [catalog]);

    // Filter metrics based on search and category
    const filteredMetrics = useMemo(() => {
        return Object.entries(catalog).filter(([id, metric]) => {
            const matchesSearch =
                id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                metric.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                metric.description.toLowerCase().includes(searchTerm.toLowerCase());

            let matchesCategory = true;
            switch (categoryFilter) {
                case 'No-Reference (NR) Metrics':
                    matchesCategory = metric.type === 'nr';
                    break;
                case 'Full-Reference (FR) Metrics':
                    matchesCategory = metric.type === 'fr';
                    break;
                case 'Selected':
                    matchesCategory = selected[metric.type].includes(id);
                    break;
                case 'Unselected':
                    matchesCategory = !selected[metric.type].includes(id);
                    break;
                case 'all':
                    matchesCategory = true;
                    break;
                default:
                    matchesCategory = metric.category === categoryFilter;
            }

            return matchesSearch && matchesCategory;
        });
    }, [catalog, searchTerm, categoryFilter, selected]);

    // Load catalog and selected metrics
    useEffect(() => {
        const fetchData = async () => {
            try {
                const [catalogRes, selectedRes] = await Promise.all([
                    fetch('http://localhost:5000/metrics/catalog'),
                    fetch('http://localhost:5000/metrics/selected')
                ]);

                const catalogData = await catalogRes.json();
                const selectedData = await selectedRes.json();

                setCatalog(catalogData);
                setSelected(selectedData);
                setLoading(false);
            } catch (error) {
                console.error('Error loading metrics:', error);
                setLoading(false);
            }
        };

        if (isOpen) {
            fetchData();
        }
    }, [isOpen]);

    const handleMetricToggle = (metricId: string, metricType: 'nr' | 'fr') => {
        setSelected(prev => {
            const newSelected = { ...prev };
            const typeArray = newSelected[metricType];

            if (typeArray.includes(metricId)) {
                newSelected[metricType] = typeArray.filter(id => id !== metricId);
            } else {
                newSelected[metricType] = [...typeArray, metricId];
            }

            return newSelected;
        });
    };

    const handleSave = async () => {
        try {
            await fetch('http://localhost:5000/metrics/selected', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(selected)
            });
            onClose();
        } catch (error) {
            console.error('Error saving metrics:', error);
        }
    };

    // Add handlers for Select All and Deselect All
    const handleSelectAll = () => {
        const metricsToSelect = filteredMetrics.reduce((acc, [id, metric]) => {
            if (!selected[metric.type].includes(id)) {
                acc[metric.type].push(id);
            }
            return acc;
        }, { nr: [...selected.nr], fr: [...selected.fr] });
        
        setSelected(metricsToSelect);
    };

    const handleDeselectAll = () => {
        const metricsToKeep = { 
            nr: selected.nr.filter(id => !filteredMetrics.some(([filteredId]) => filteredId === id)),
            fr: selected.fr.filter(id => !filteredMetrics.some(([filteredId]) => filteredId === id))
        };
        
        setSelected(metricsToKeep);
    };

    return (
        <>
            {isOpen && (
                <div className="metrics-modal-overlay">
                    <div className="metrics-config-modal">
                        <div className="metrics-config-content">
                            <div className="metrics-filters">
                                <input
                                    type="text"
                                    placeholder="Search metrics..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="metrics-search"
                                />
                                <select
                                    value={categoryFilter}
                                    onChange={(e) => setCategoryFilter(e.target.value)}
                                    className="metrics-category-filter"
                                >
                                    {categories.map(category => (
                                        <option key={category} value={category}>
                                            {category === 'all' ? 'All Categories' : category}
                                        </option>
                                    ))}
                                </select>
                                <div className="metrics-bulk-actions">
                                    <button 
                                        onClick={handleSelectAll}
                                        className="select-all-button"
                                    >
                                        Select All
                                    </button>
                                    <button 
                                        onClick={handleDeselectAll}
                                        className="deselect-all-button"
                                    >
                                        Deselect All
                                    </button>
                                </div>
                            </div>

                            {loading ? (
                                <div className="metrics-loading">Loading metrics...</div>
                            ) : (
                                <div className="metrics-list">
                                    <div className="metrics-group">
                                        <h3>No-Reference Metrics</h3>
                                        {filteredMetrics
                                            .filter(([, metric]) => metric.type === 'nr')
                                            .map(([id, metric]) => (
                                                <div key={id} className="metric-item">
                                                    <label>
                                                        <input
                                                            type="checkbox"
                                                            checked={selected.nr.includes(id)}
                                                            onChange={() => handleMetricToggle(id, 'nr')}
                                                        />
                                                        <div className="metric-info">
                                                            <strong>
                                                                {metric.name} <span className="metric-id">({id})</span>
                                                            </strong>
                                                            <p>{metric.description}</p>
                                                            <span className="metric-details">
                                                                Range: {metric.score_range.join(' - ')}
                                                                ({metric.higher_better ? 'Higher is better' : 'Lower is better'})
                                                            </span>
                                                        </div>
                                                    </label>
                                                </div>
                                            ))}
                                    </div>

                                    <div className="metrics-group">
                                        <h3>Full-Reference Metrics</h3>
                                        {filteredMetrics
                                            .filter(([, metric]) => metric.type === 'fr')
                                            .map(([id, metric]) => (
                                                <div key={id} className="metric-item">
                                                    <label>
                                                        <input
                                                            type="checkbox"
                                                            checked={selected.fr.includes(id)}
                                                            onChange={() => handleMetricToggle(id, 'fr')}
                                                        />
                                                        <div className="metric-info">
                                                            <strong>
                                                                {metric.name} <span className="metric-id">({id})</span>
                                                            </strong>
                                                            <p>{metric.description}</p>
                                                            <span className="metric-details">
                                                                Range: {metric.score_range.join(' - ')}
                                                                ({metric.higher_better ? 'Higher is better' : 'Lower is better'})
                                                            </span>
                                                        </div>
                                                    </label>
                                                </div>
                                            ))}
                                    </div>
                                </div>
                            )}

                            <div className="metrics-actions">
                                <button onClick={handleSave} className="save-button">
                                    Save Changes
                                </button>
                                <button onClick={onClose} className="cancel-button">
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};