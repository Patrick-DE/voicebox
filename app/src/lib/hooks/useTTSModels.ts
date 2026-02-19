import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

export interface TTSModelOption {
  /** Value sent to the backend as `model_size` in GenerationRequest. */
  value: string;
  /** Human-readable label shown in the dropdown. */
  label: string;
  /** Whether the model is currently downloaded (ready to use). */
  downloaded: boolean;
  /** Whether this is a user-added custom model. */
  isCustom: boolean;
}

/**
 * Returns the list of TTS models (built-in Qwen + custom) with their download status.
 * Use `downloaded` to filter to only models that are ready to use.
 */
export function useTTSModels() {
  const { data: modelStatus, isLoading } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: () => apiClient.getModelStatus(),
    staleTime: 30_000,
  });

  const models: TTSModelOption[] = (modelStatus?.models ?? [])
    .filter((m) => m.model_name.startsWith('qwen-tts') || m.is_custom)
    .map((m) => {
      // Built-in Qwen models: strip the "qwen-tts-" prefix to get "1.7B" / "0.6B"
      const value = m.model_name.startsWith('qwen-tts')
        ? m.model_name.replace('qwen-tts-', '')
        : m.model_name; // custom models: use the full model_name as-is
      return {
        value,
        label: m.display_name,
        downloaded: m.downloaded,
        isCustom: !!m.is_custom,
      };
    });

  return { models, isLoading };
}
