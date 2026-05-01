/**
 * ResultCard — Displays CNN prediction results with confidence bar,
 * risk level badge, and full probability distribution.
 */

import React from 'react';
import { AlertTriangle, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { PredictionResponse } from '../api/triageApi';

interface ResultCardProps {
  result: PredictionResponse;
}

const RISK_CONFIG = {
  none: { color: 'text-green-600', bg: 'bg-green-50 border-green-200', icon: CheckCircle, label: 'No Risk' },
  low: { color: 'text-blue-600', bg: 'bg-blue-50 border-blue-200', icon: CheckCircle, label: 'Low Risk' },
  medium: { color: 'text-yellow-600', bg: 'bg-yellow-50 border-yellow-200', icon: AlertCircle, label: 'Medium Risk' },
  high: { color: 'text-red-600', bg: 'bg-red-50 border-red-200', icon: AlertTriangle, label: 'High Risk — Refer Immediately' },
} as const;

const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  const risk = result.risk_level as keyof typeof RISK_CONFIG;
  const riskConfig = RISK_CONFIG[risk] || RISK_CONFIG.low;
  const RiskIcon = riskConfig.icon;

  return (
    <div className="space-y-4">
      {/* Primary result */}
      <div className={`border-2 rounded-2xl p-6 ${riskConfig.bg}`}>
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-xl font-bold text-gray-900">{result.predicted_class}</h3>
            <p className={`text-sm font-semibold mt-1 flex items-center gap-1 ${riskConfig.color}`}>
              <RiskIcon size={16} />
              {riskConfig.label}
            </p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold text-gray-900">
              {(result.confidence * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-gray-500">Confidence</p>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-700 ${
                result.confidence > 0.8 ? 'bg-green-500' :
                result.confidence > 0.6 ? 'bg-blue-500' :
                result.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
            />
          </div>
        </div>

        {/* Inference time */}
        <p className="mt-2 text-xs text-gray-400 flex items-center gap-1">
          <Clock size={12} />
          Inference time: {result.inference_time_ms}ms
        </p>
      </div>

      {/* Low confidence warning */}
      {result.low_confidence_warning && (
        <div className="flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-xl text-yellow-700 text-sm">
          <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" />
          <p>Low confidence prediction. Consider retaking the photo with better lighting and focus.</p>
        </div>
      )}

      {/* All probabilities */}
      <div className="bg-white border rounded-2xl p-4">
        <h4 className="font-semibold text-gray-700 mb-3 text-sm uppercase tracking-wide">
          All Classes
        </h4>
        <div className="space-y-2">
          {result.all_probabilities.map((item) => (
            <div key={item.class} className="flex items-center gap-2">
              <div className="w-36 text-xs text-gray-600 truncate flex-shrink-0">{item.class}</div>
              <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                <div
                  className="h-1.5 rounded-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${(item.probability * 100).toFixed(1)}%` }}
                />
              </div>
              <div className="w-12 text-xs text-right text-gray-500">
                {(item.probability * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
