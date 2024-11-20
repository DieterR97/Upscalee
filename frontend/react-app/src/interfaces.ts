export interface SpandrelInfo {
  architecture: string;
  is_supported: boolean;
  input_channels: number;
  output_channels: number;
  scale: number;
  supports_half: boolean;
  supports_bfloat16: boolean;
  size_requirements: any;
  tiling: string;
}

export interface UnregisteredModel {
  name: string;
  file_name: string;
  file_pattern: string;
  scale: number | null;
  path: string;
  spandrel_info?: SpandrelInfo;
} 