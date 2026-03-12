import React, { Component, type ReactNode } from 'react';
import { AlertCircle } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render(): React.ReactNode {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-[#121212] flex items-center justify-center p-6">
          <div className="bg-[#282828] rounded-xl p-8 max-w-md text-center">
            <AlertCircle className="w-12 h-12 text-[#E34040] mx-auto mb-4" />
            <h2 className="text-xl font-bold text-white mb-2">Something went wrong</h2>
            <p className="text-[#B3B3B3] mb-6">{this.state.error.message}</p>
            <button
              onClick={() => this.setState({ error: null })}
              className="bg-[#5B8DEF] hover:bg-[#7BA4F7] text-black font-bold py-3 px-6 rounded-full transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
