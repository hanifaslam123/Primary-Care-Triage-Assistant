/**
 * Triage API client — communicates with the FastAPI backend.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30_000,
});

// ---- Types ----

export interface ClassProbability {
  class: string;
  probability: number;
  risk: string;
}

export interface PredictionResponse {
  predicted_class: string;
  predicted_class_index: number;
  confidence: number;
  risk_level: string;
  all_probabilities: ClassProbability[];
  inference_time_ms: number;
  low_confidence_warning: boolean;
  disclaimer: string;
}

export interface ModelInfo {
  architecture: string;
  accuracy: number;
  classes: string[];
  input_size: number[];
  training_samples: number;
}

// ---- API Functions ----

/**
 * Upload a skin image and receive classification results.
 * Real-time inference — typical response under 1 second.
 */
export async function classifySkinImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<PredictionResponse>('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
}

/** Fetch model metadata. */
export async function getModelInfo(): Promise<ModelInfo> {
  const response = await api.get<ModelInfo>('/model/info');
  return response.data;
}

/** Check API health. */
export async function checkHealth(): Promise<{ status: string }> {
  const response = await api.get('/health');
  return response.data;
}
