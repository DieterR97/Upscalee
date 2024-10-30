export interface QueuedImage {
    originalImage: string;
    upscaledImage: string;
    modelName: string;
    scale: number;
    originalFilename: string;
}

export interface MetricInfo {
    name: string;
    description: string;
    score_range: [number, number];
    type: 'fr' | 'nr';
    higher_better: boolean;
}
