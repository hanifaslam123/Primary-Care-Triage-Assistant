/**
 * ImageUploader — Drag & drop or click-to-upload skin image component.
 * Accepts JPG, PNG, WebP. Max file size: 10MB.
 */

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image, X } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelected: (file: File, preview: string) => void;
  isLoading: boolean;
}

const ACCEPTED_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/webp': ['.webp'],
};

const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected, isLoading }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: unknown[]) => {
      setError(null);

      if (rejectedFiles.length > 0) {
        setError('Invalid file. Please upload a JPG, PNG, or WebP image under 10MB.');
        return;
      }

      if (acceptedFiles.length === 0) return;

      const file = acceptedFiles[0];
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
      onImageSelected(file, objectUrl);
    },
    [onImageSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
    disabled: isLoading,
  });

  const clearPreview = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreview(null);
    setError(null);
  };

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer
          transition-all duration-200
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'}
          ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />

        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt="Uploaded skin image"
              className="max-h-64 mx-auto rounded-xl object-contain"
            />
            {!isLoading && (
              <button
                onClick={clearPreview}
                className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors"
                aria-label="Remove image"
              >
                <X size={16} />
              </button>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-gray-500">
            {isDragActive ? (
              <>
                <Image size={48} className="text-blue-500" />
                <p className="text-blue-600 font-medium">Drop the image here</p>
              </>
            ) : (
              <>
                <Upload size={48} className="text-gray-400" />
                <p className="text-lg font-medium">
                  Drag & drop a skin image, or <span className="text-blue-600 underline">click to upload</span>
                </p>
                <p className="text-sm text-gray-400">JPG, PNG, or WebP — max 10 MB</p>
              </>
            )}
          </div>
        )}
      </div>

      {error && (
        <p className="mt-2 text-sm text-red-600 flex items-center gap-1">
          <X size={14} />
          {error}
        </p>
      )}
    </div>
  );
};

export default ImageUploader;
