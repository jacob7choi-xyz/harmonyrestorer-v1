import React, { Component, type ReactNode } from 'react';
import { AlertCircle } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
  resetKey: number;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { error: null, resetKey: 0 };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  render(): React.ReactNode {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-base flex items-center justify-center p-6">
          <div className="bg-card backdrop-blur-md border border-glass rounded-lg p-8 max-w-md text-center">
            <AlertCircle className="w-12 h-12 text-destructive mx-auto mb-4" />
            <h2 className="text-xl font-bold text-ink mb-2">Something went wrong</h2>
            <p className="text-ink-secondary mb-6">{this.state.error.message}</p>
            <button
              onClick={() => this.setState(s => ({ error: null, resetKey: s.resetKey + 1 }))}
              className="bg-amber hover:bg-amber-deep text-on-amber font-bold py-3 px-6 rounded-full transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return <div key={this.state.resetKey}>{this.props.children}</div>;
  }
}
