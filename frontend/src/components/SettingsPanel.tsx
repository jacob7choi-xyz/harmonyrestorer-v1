import { Settings } from 'lucide-react';
import type { ProcessingSettings } from '../types';

interface SettingsPanelProps {
  settings: ProcessingSettings;
  onSettingsChange: (settings: ProcessingSettings) => void;
  disabled: boolean;
}

export function SettingsPanel({ settings, onSettingsChange, disabled }: SettingsPanelProps) {
  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/20 shadow-lg overflow-hidden">
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
            <Settings className="w-4 h-4 text-white/70" />
          </div>
          <h3 className="text-lg font-semibold text-white/90">Settings</h3>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Noise Reduction */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-3">Noise Reduction</label>
          <div className="grid grid-cols-2 gap-2">
            {(['light', 'medium', 'strong', 'extreme'] as const).map((level) => (
              <button
                key={level}
                onClick={() => onSettingsChange({ ...settings, noise_reduction: level })}
                disabled={disabled}
                className={`px-3 py-2.5 text-sm rounded-xl transition-all font-medium capitalize ${
                  settings.noise_reduction === level
                    ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-white/10 text-white/70 hover:bg-white/20 border border-white/20'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {level}
              </button>
            ))}
          </div>
        </div>

        {/* Toggle Options */}
        <div className="space-y-3">
          {([
            { key: 'enhance_speech' as const, label: 'Speech Enhancement' },
            { key: 'remove_reverb' as const, label: 'Remove Reverb' },
            { key: 'isolate_voice' as const, label: 'Voice Isolation' },
            { key: 'boost_clarity' as const, label: 'Boost Clarity' },
          ]).map(({ key, label }) => (
            <label
              key={key}
              className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/10 cursor-pointer hover:bg-white/10 transition-all"
            >
              <span className="text-sm font-medium text-white/80">{label}</span>
              <div className="relative">
                <input
                  type="checkbox"
                  checked={settings[key]}
                  onChange={(e) => onSettingsChange({ ...settings, [key]: e.target.checked })}
                  disabled={disabled}
                  className="sr-only"
                />
                <div className={`w-11 h-6 rounded-full transition-all ${settings[key] ? 'bg-blue-500' : 'bg-white/20'}`}>
                  <div
                    className={`w-5 h-5 rounded-full bg-white transition-all duration-200 ease-out ${
                      settings[key] ? 'translate-x-5' : 'translate-x-0.5'
                    } mt-0.5 shadow-lg`}
                  />
                </div>
              </div>
            </label>
          ))}
        </div>

        {/* Output Format */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-3">Output Format</label>
          <div className="relative">
            <select
              value={settings.output_format}
              onChange={(e) => onSettingsChange({ ...settings, output_format: e.target.value as ProcessingSettings['output_format'] })}
              disabled={disabled}
              className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white/90 focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none backdrop-blur-sm"
            >
              <option value="wav" className="bg-gray-800">WAV (Lossless)</option>
              <option value="flac" className="bg-gray-800">FLAC (Compressed)</option>
              <option value="aiff" className="bg-gray-800">AIFF (Professional)</option>
            </select>
            <div className="absolute right-3 top-3 pointer-events-none">
              <svg className="w-5 h-5 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>

        {/* Quality */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-3">Quality</label>
          <div className="grid grid-cols-3 gap-2">
            {(['standard', 'high', 'pro'] as const).map((quality) => (
              <button
                key={quality}
                onClick={() => onSettingsChange({ ...settings, quality })}
                disabled={disabled}
                className={`px-3 py-2.5 text-sm rounded-xl transition-all font-medium capitalize ${
                  settings.quality === quality
                    ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-white/10 text-white/70 hover:bg-white/20 border border-white/20'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {quality}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
