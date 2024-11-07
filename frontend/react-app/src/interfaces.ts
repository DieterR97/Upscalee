export interface ModelInfo {
  name: string;
  description: string;
  scale: number;
  variable_scale: boolean;
  type: string;
  file_pattern: string;
}

export interface UnregisteredModel {
  file_name: string;
  path: string;
}

export interface ModelsResponse {
  registered: Record<string, ModelInfo>;
  custom: Record<string, ModelInfo>;
  unregistered: UnregisteredModel[];
} 