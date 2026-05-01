/**
 * Home page — main triage interface.
 * Upload a skin image, click Analyze, see real-time CNN results.
 */

import React, { useState } from 'react';
import { Stethoscope, Loader2 } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import ResultCard from '../components/ResultCard';
import { classifySkinImage, PredictionResponse } from '../api/triageApi';

const Home: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelected = (file: File, previewUrl: string) => {
    setSelectedFile(file);
    setPreview(previewUrl);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await classifySkinImage(selectedFile);
      setResult(prediction);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'An unexpected error occurred.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-xl">
            <Stethoscope size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">Primary Care Triage Assistant</h1>
            <p className="text-sm text-gray-500">AI-powered skin anomaly analysis — for clinical use only</p>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Medical disclaimer banner */}
        <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 mb-6 text-amber-800 text-sm">
          <strong>Medical Disclaimer:</strong> This tool is intended to <em>assist</em> healthcare
          professionals and is <strong>NOT</strong> a substitute for professional medical diagnosis.
          All results must be reviewed by a licensed healthcare provider.
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left panel — upload */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
            <h2 className="text-lg font-semibold text-gray-800">Upload Image</h2>
            <ImageUploader onImageSelected={handleImageSelected} isLoading={isLoading} />

            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || isLoading}
              className={`w-full py-3 px-6 rounded-xl font-semibold text-white transition-all duration-200 flex items-center justify-center gap-2
                ${selectedFile && !isLoading
                  ? 'bg-blue-600 hover:bg-blue-700 shadow-md hover:shadow-lg'
                  : 'bg-gray-300 cursor-not-allowed'
                }`}
            >
              {isLoading ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                'Analyze Image'
              )}
            </button>

            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
                {error}
              </div>
            )}
          </div>

          {/* Right panel — results */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Analysis Results</h2>
            {result ? (
              <ResultCard result={result} />
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-gray-400 text-center space-y-2">
                <Stethoscope size={48} className="opacity-30" />
                <p>Upload an image and click <strong>Analyze</strong> to see results.</p>
              </div>
            )}
          </div>
        </div>

        {/* Model info footer */}
        <div className="mt-8 text-center text-xs text-gray-400 space-y-1">
          <p>ResNet-50 CNN trained on 5,000+ clinical images · 85% classification accuracy</p>
          <p>Real-time inference — typical response under 1 second</p>
        </div>
      </main>
    </div>
  );
};

export default Home;
