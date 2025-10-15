import { useEffect, useRef, useState } from 'react';

interface UsePollingOptions {
  interval?: number;
  enabled?: boolean;
  onError?: (error: Error) => void;
}

export function usePolling<T>(
  fetchFunction: () => Promise<T>,
  options: UsePollingOptions = {}
) {
  const { interval = 5000, enabled = true, onError } = options;
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const intervalRef = useRef<number | null>(null);
  const isMountedRef = useRef(true);

  const fetchData = async () => {
    if (!isMountedRef.current) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await fetchFunction();
      if (isMountedRef.current) {
        setData(result);
      }
    } catch (err) {
      if (isMountedRef.current) {
        const error = err instanceof Error ? err : new Error('Unknown error');
        setError(error);
        onError?.(error);
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  };

  const startPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    // Fetch immediately
    fetchData();
    
    // Then poll at intervals
    if (enabled) {
      intervalRef.current = setInterval(fetchData, interval);
    }
  };

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const refresh = () => {
    fetchData();
  };

  useEffect(() => {
    isMountedRef.current = true;
    
    if (enabled) {
      startPolling();
    } else {
      stopPolling();
    }

    return () => {
      isMountedRef.current = false;
      stopPolling();
    };
  }, [enabled, interval]);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      stopPolling();
    };
  }, []);

  return {
    data,
    loading,
    error,
    refresh,
    startPolling,
    stopPolling
  };
}
