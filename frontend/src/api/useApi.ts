import { useEffect, useRef, useState, type DependencyList } from "react";
import { ApiError } from "./client";

export type AsyncState<T> =
  | { status: "loading"; data: undefined; error: undefined }
  | { status: "success"; data: T; error: undefined }
  | { status: "error"; data: undefined; error: ApiError };

/**
 * Runs `fetcher` whenever `deps` change, exposing a small state machine
 * (loading / success / error) instead of ad-hoc booleans in every component.
 * Requests are aborted on cleanup/re-run so stale responses never overwrite
 * fresher ones.
 */
export function useAsync<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: DependencyList,
): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({
    status: "loading",
    data: undefined,
    error: undefined,
  });
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading", data: undefined, error: undefined });

    fetcherRef
      .current(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setState({ status: "success", data, error: undefined });
        }
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const apiError =
          err instanceof ApiError
            ? err
            : new ApiError(0, "Unexpected error while contacting the API.");
        setState({ status: "error", data: undefined, error: apiError });
      });

    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
