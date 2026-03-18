/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { UploadCloud, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ status: 'fractured' | 'normal' | 'error', message: string, confidence?: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setResult({ status: 'error', message: 'Please upload a valid image file.' });
      return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setResult(null);

    // SIMULATION for AI Studio Preview
    // In your real Flask app, this is handled by the fetch('/predict') call in script.js
    setTimeout(() => {
      const isFractured = Math.random() > 0.5;
      setResult({
        status: isFractured ? 'fractured' : 'normal',
        message: isFractured ? 'FRACTURED BONE DETECTED' : 'NORMAL BONE',
        confidence: isFractured ? 0.85 + Math.random() * 0.14 : 0.1 + Math.random() * 0.3
      });
      setLoading(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4 font-sans text-gray-900">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-8 text-center">
        <header className="mb-8">
          <h1 className="text-2xl font-bold mb-2">🦴 Bone Fracture Detection</h1>
          <p className="text-gray-500 text-sm">Upload an X-ray image to detect Fractured or Normal bone.</p>
        </header>

        <form onSubmit={handleSubmit}>
          <div
            className={`border-2 border-dashed rounded-xl p-8 mb-6 cursor-pointer transition-colors relative ${preview ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 hover:border-indigo-400 bg-gray-50 hover:bg-indigo-50/50'}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/jpeg, image/png, image/jpg"
              onChange={(e) => e.target.files && handleFile(e.target.files[0])}
            />

            {preview ? (
              <img src={preview} alt="Preview" className="max-h-48 mx-auto rounded-lg shadow-sm" />
            ) : (
              <div className="flex flex-col items-center text-gray-500">
                <UploadCloud className="w-12 h-12 mb-3 text-indigo-400" />
                <p className="font-medium mb-1">Click to browse or drag & drop</p>
                <p className="text-xs">Supports JPG, JPEG, PNG</p>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={!file || loading}
            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-semibold py-3 px-4 rounded-xl transition-colors flex justify-center items-center"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin mr-2" />
                Analyzing...
              </>
            ) : (
              'Analyze X-ray'
            )}
          </button>
        </form>

        {result && (
          <div className={`mt-6 p-4 rounded-xl border animate-in fade-in slide-in-from-bottom-2 ${
            result.status === 'fractured' ? 'bg-red-50 border-red-200 text-red-700' :
            result.status === 'normal' ? 'bg-emerald-50 border-emerald-200 text-emerald-700' :
            'bg-amber-50 border-amber-200 text-amber-700'
          }`}>
            <div className="flex items-center justify-center mb-1">
              {result.status === 'fractured' ? <AlertCircle className="w-5 h-5 mr-2" /> :
               result.status === 'normal' ? <CheckCircle className="w-5 h-5 mr-2" /> :
               <AlertCircle className="w-5 h-5 mr-2" />}
              <h2 className="text-lg font-bold">{result.message}</h2>
            </div>
            {result.confidence && (
              <p className="text-sm opacity-90">Confidence Score: {result.confidence.toFixed(2)}</p>
            )}
          </div>
        )}
        
        <div className="mt-8 pt-6 border-t border-gray-100 text-xs text-gray-400">
          <p><strong>Note:</strong> This live preview uses a simulated frontend.</p>
          <p>Download the generated Flask code to run your real ML model.</p>
        </div>
      </div>
    </div>
  );
}
