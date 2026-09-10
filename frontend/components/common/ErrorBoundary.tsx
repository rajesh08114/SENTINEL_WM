"use client";
import * as React from "react";

interface State {
  error: Error | null;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  State
> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("view error:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="m-6 rounded-lg border border-danger/40 bg-danger/10 p-6 text-sm text-danger">
          <p className="font-semibold">This view crashed.</p>
          <p className="mt-1 font-mono text-xs opacity-80">
            {this.state.error.message}
          </p>
          <button
            className="mt-3 rounded border border-danger px-3 py-1 text-xs hover:bg-danger hover:text-black"
            onClick={() => this.setState({ error: null })}
          >
            retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
