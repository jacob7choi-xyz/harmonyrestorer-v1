import { Upload, Brain, Loader2, CheckCircle, AlertCircle, Activity, TrendingUp } from 'lucide-react';
import type { ProcessingStatus } from '../types';

interface ProgressCardProps {
  status: ProcessingStatus;
}

function StatusIcon({ current }: { current: ProcessingStatus['status'] }) {
  switch (current) {
    case 'uploading':
      return <Upload className="w-5 h-5 text-blue-500" />;
    case 'queued':
      return <Brain className="w-5 h-5 text-purple-500" />;
    case 'processing':
      return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />;
    case 'completed':
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    case 'failed':
      return <AlertCircle className="w-5 h-5 text-red-500" />;
    default:
      return <Activity className="w-5 h-5 text-white/50" />;
  }
}

function barColor(current: ProcessingStatus['status']): string {
  switch (current) {
    case 'processing':
      return 'from-blue-500 to-blue-400';
    case 'completed':
      return 'from-green-500 to-green-400';
    case 'failed':
      return 'from-red-500 to-red-400';
    default:
      return 'from-white/30 to-white/20';
  }
}

export function ProgressCard({ status }: ProgressCardProps) {
  const showBar = status.status === 'processing' || status.status === 'uploading' || status.status === 'queued';

  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-lg">
      <div className="flex items-center space-x-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center border border-white/20">
          <StatusIcon current={status.status} />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white/90 capitalize">
            {status.status === 'idle' ? 'Ready' : status.status}
          </h3>
          <p className="text-sm text-white/60">{status.message}</p>
        </div>
      </div>

      {showBar && (
        <div className="mb-4">
          <div className="flex justify-between text-sm text-white/60 mb-2">
            <span>Progress</span>
            <span>{Math.round(status.progress)}%</span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden backdrop-blur-sm">
            <div
              className={`h-full bg-gradient-to-r ${barColor(status.status)} transition-all duration-500 ease-out rounded-full`}
              style={{ width: `${status.progress}%` }}
            />
          </div>
        </div>
      )}

      {status.status === 'completed' && status.processingTime != null && (
        <div className="bg-green-500/10 border border-green-400/20 rounded-2xl p-4 mt-4 backdrop-blur-sm">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-green-400 font-medium">
              Processed in {status.processingTime.toFixed(1)}s
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
