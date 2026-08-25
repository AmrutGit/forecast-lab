import type { ReactNode } from "react";
import { ApiError } from "../api/client";

export function LoadingBlock({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="status-block status-block--loading" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function Skeleton({ height = 220 }: { height?: number }) {
  return (
    <div
      className="skeleton"
      style={{ height }}
      role="status"
      aria-label="Loading content"
    />
  );
}

export function ErrorBlock({
  error,
  onRetry,
}: {
  error: ApiError | Error;
  onRetry?: () => void;
}) {
  const isApiError = error instanceof ApiError;
  const isNetwork = isApiError && error.status === 0;

  return (
    <div className="status-block status-block--error" role="alert">
      <span className="status-icon" aria-hidden="true">
        !
      </span>
      <div className="status-block__body">
        <p className="status-block__title">
          {isNetwork ? "Cannot reach the backend" : "Something went wrong"}
        </p>
        <p className="status-block__detail">{error.message}</p>
      </div>
      {onRetry && (
        <button type="button" className="btn btn--ghost" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyBlock({ children }: { children: ReactNode }) {
  return (
    <div className="status-block status-block--empty">
      <span className="status-icon status-icon--muted" aria-hidden="true">
        ○
      </span>
      <p className="status-block__detail">{children}</p>
    </div>
  );
}
