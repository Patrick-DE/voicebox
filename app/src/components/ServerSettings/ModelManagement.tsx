import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Download, Loader2, Plus, Trash2 } from 'lucide-react';
import { useCallback, useState } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';

export function ModelManagement() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);
  const [customModelUrl, setCustomModelUrl] = useState('');
  const [customModelDisplayName, setCustomModelDisplayName] = useState('');

  const { data: modelStatus, isLoading } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: async () => {
      console.log('[Query] Fetching model status');
      const result = await apiClient.getModelStatus();
      console.log('[Query] Model status fetched:', result);
      return result;
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Callbacks for download completion
  const handleDownloadComplete = useCallback(() => {
    console.log('[ModelManagement] Download complete, clearing state');
    setDownloadingModel(null);
    setDownloadingDisplayName(null);
    queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
  }, [queryClient]);

  const handleDownloadError = useCallback(() => {
    console.log('[ModelManagement] Download error, clearing state');
    setDownloadingModel(null);
    setDownloadingDisplayName(null);
  }, []);

  // Use progress toast hook for the downloading model
  useModelDownloadToast({
    modelName: downloadingModel || '',
    displayName: downloadingDisplayName || '',
    enabled: !!downloadingModel && !!downloadingDisplayName,
    onComplete: handleDownloadComplete,
    onError: handleDownloadError,
  });

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<{
    name: string;
    displayName: string;
    sizeMb?: number;
    isCustom?: boolean;
  } | null>(null);

  const handleDownload = async (modelName: string) => {
    console.log('[Download] Button clicked for:', modelName, 'at', new Date().toISOString());
    
    // Find display name
    const model = modelStatus?.models.find((m) => m.model_name === modelName);
    const displayName = model?.display_name || modelName;
    
    try {
      // IMPORTANT: Call the API FIRST before setting state
      // Setting state enables the SSE EventSource in useModelDownloadToast,
      // which can block/delay the download fetch due to HTTP/1.1 connection limits
      console.log('[Download] Calling download API for:', modelName);
      const result = await apiClient.triggerModelDownload(modelName);
      console.log('[Download] Download API responded:', result);
      
      // NOW set state to enable SSE tracking (after download has started on backend)
      setDownloadingModel(modelName);
      setDownloadingDisplayName(displayName);
      
      // Download initiated successfully - state will be cleared when SSE reports completion
      // or by the polling interval detecting the model is downloaded
      queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    } catch (error) {
      console.error('[Download] Download failed:', error);
      setDownloadingModel(null);
      setDownloadingDisplayName(null);
      toast({
        title: 'Download failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  };

  const deleteMutation = useMutation({
    mutationFn: async (modelName: string) => {
      console.log('[Delete] Deleting model:', modelName);
      const result = await apiClient.deleteModel(modelName);
      console.log('[Delete] Model deleted successfully:', modelName);
      return result;
    },
    onSuccess: async (_data, _modelName) => {
      console.log('[Delete] onSuccess - showing toast and invalidating queries');
      toast({
        title: 'Model deleted',
        description: `${modelToDelete?.displayName || 'Model'} has been deleted successfully.`,
      });
      setDeleteDialogOpen(false);
      setModelToDelete(null);
      // Invalidate AND explicitly refetch to ensure UI updates
      // Using refetchType: 'all' ensures we refetch even if the query is stale
      console.log('[Delete] Invalidating modelStatus query');
      await queryClient.invalidateQueries({ 
        queryKey: ['modelStatus'],
        refetchType: 'all',
      });
      // Also explicitly refetch to guarantee fresh data
      console.log('[Delete] Explicitly refetching modelStatus query');
      await queryClient.refetchQueries({ queryKey: ['modelStatus'] });
      console.log('[Delete] Query refetched');
    },
    onError: (error: Error) => {
      console.log('[Delete] onError:', error);
      toast({
        title: 'Delete failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const addCustomModelMutation = useMutation({
    mutationFn: async ({ url, displayName }: { url: string; displayName?: string }) => {
      return apiClient.addCustomModel(url, displayName || undefined);
    },
    onSuccess: async (data) => {
      toast({
        title: 'Model added',
        description: `${data.display_name} has been added to your custom models.`,
      });
      setCustomModelUrl('');
      setCustomModelDisplayName('');
      await queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to add model',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const handleAddCustomModel = () => {
    if (!customModelUrl.trim()) return;
    addCustomModelMutation.mutate({
      url: customModelUrl.trim(),
      displayName: customModelDisplayName.trim() || undefined,
    });
  };

  const formatSize = (sizeMb?: number): string => {
    if (!sizeMb) return 'Unknown';
    if (sizeMb < 1024) return `${sizeMb.toFixed(1)} MB`;
    return `${(sizeMb / 1024).toFixed(2)} GB`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Management</CardTitle>
        <CardDescription>
          Download and manage AI models for voice generation and transcription
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : modelStatus ? (
          <div className="space-y-4">
            {/* TTS Models */}
            <div>
              <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
                Voice Generation Models
              </h3>
              <div className="space-y-2">
                {modelStatus.models
                  .filter((m) => m.model_name.startsWith('qwen-tts'))
                  .map((model) => (
                    <ModelItem
                      key={model.model_name}
                      model={model}
                      onDownload={() => handleDownload(model.model_name)}
                      onDelete={() => {
                        setModelToDelete({
                          name: model.model_name,
                          displayName: model.display_name,
                          sizeMb: model.size_mb,
                        });
                        setDeleteDialogOpen(true);
                      }}
                      isDownloading={downloadingModel === model.model_name}
                      formatSize={formatSize}
                    />
                  ))}
              </div>
            </div>

            {/* Whisper Models */}
            <div>
              <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
                Transcription Models
              </h3>
              <div className="space-y-2">
                {modelStatus.models
                  .filter((m) => m.model_name.startsWith('whisper'))
                  .map((model) => (
                    <ModelItem
                      key={model.model_name}
                      model={model}
                      onDownload={() => handleDownload(model.model_name)}
                      onDelete={() => {
                        setModelToDelete({
                          name: model.model_name,
                          displayName: model.display_name,
                          sizeMb: model.size_mb,
                        });
                        setDeleteDialogOpen(true);
                      }}
                      isDownloading={downloadingModel === model.model_name}
                      formatSize={formatSize}
                    />
                  ))}
              </div>
            </div>

            {/* Custom Models */}
            <div>
              <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
                Custom Models
              </h3>
              <div className="space-y-2">
                {modelStatus.models
                  .filter((m) => m.is_custom)
                  .map((model) => (
                    <ModelItem
                      key={model.model_name}
                      model={model}
                      onDownload={() => handleDownload(model.model_name)}
                      onDelete={() => {
                        setModelToDelete({
                          name: model.model_name,
                          displayName: model.display_name,
                          sizeMb: model.size_mb,
                          isCustom: true,
                        });
                        setDeleteDialogOpen(true);
                      }}
                      isDownloading={downloadingModel === model.model_name}
                      formatSize={formatSize}
                    />
                  ))}

                {/* Add custom model form */}
                <div className="pt-2 space-y-2">
                  <Input
                    placeholder="HuggingFace URL or repo ID (e.g., hexgrad/Kokoro-82M)"
                    value={customModelUrl}
                    onChange={(e) => setCustomModelUrl(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddCustomModel()}
                    disabled={addCustomModelMutation.isPending}
                  />
                  <div className="flex gap-2">
                    <Input
                      placeholder="Display name (optional)"
                      value={customModelDisplayName}
                      onChange={(e) => setCustomModelDisplayName(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleAddCustomModel()}
                      disabled={addCustomModelMutation.isPending}
                    />
                    <Button
                      size="sm"
                      onClick={handleAddCustomModel}
                      disabled={!customModelUrl.trim() || addCustomModelMutation.isPending}
                      variant="outline"
                    >
                      {addCustomModelMutation.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Plus className="h-4 w-4" />
                      )}
                      <span className="ml-1">Add</span>
                    </Button>
                  </div>
                </div>
              </div>
            </div>

          </div>
        ) : null}
      </CardContent>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {modelToDelete?.isCustom ? 'Remove Custom Model' : 'Delete Model'}
            </AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to {modelToDelete?.isCustom ? 'remove' : 'delete'}{' '}
              <strong>{modelToDelete?.displayName}</strong>?
              {modelToDelete?.isCustom ? (
                <>
                  {' '}
                  This will remove it from your custom models list
                  {modelToDelete?.sizeMb && (
                    <> and free up {formatSize(modelToDelete.sizeMb)} of disk space</>
                  )}
                  .
                </>
              ) : (
                modelToDelete?.sizeMb && (
                  <>
                    {' '}
                    This will free up {formatSize(modelToDelete.sizeMb)} of disk space. The model
                    will need to be re-downloaded if you want to use it again.
                  </>
                )
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (modelToDelete) {
                  deleteMutation.mutate(modelToDelete.name);
                }
              }}
              disabled={deleteMutation.isPending}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleteMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  {modelToDelete?.isCustom ? 'Removing...' : 'Deleting...'}
                </>
              ) : modelToDelete?.isCustom ? (
                'Remove'
              ) : (
                'Delete'
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}

interface ModelItemProps {
  model: {
    model_name: string;
    display_name: string;
    downloaded: boolean;
    downloading?: boolean;  // From server - true if download in progress
    size_mb?: number;
    loaded: boolean;
  };
  onDownload: () => void;
  onDelete: () => void;
  isDownloading: boolean;  // Local state - true if user just clicked download
  formatSize: (sizeMb?: number) => string;
}

function ModelItem({ model, onDownload, onDelete, isDownloading, formatSize }: ModelItemProps) {
  // Use server's downloading state OR local state (for immediate feedback before server updates)
  const showDownloading = model.downloading || isDownloading;
  
  return (
    <div className="flex items-center justify-between p-3 border rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{model.display_name}</span>
          {model.loaded && (
            <Badge variant="default" className="text-xs">
              Loaded
            </Badge>
          )}
          {/* Only show Downloaded if actually downloaded AND not downloading */}
          {model.downloaded && !model.loaded && !showDownloading && (
            <Badge variant="secondary" className="text-xs">
              Downloaded
            </Badge>
          )}
        </div>
        {model.downloaded && model.size_mb && !showDownloading && (
          <div className="text-xs text-muted-foreground mt-1">
            Size: {formatSize(model.size_mb)}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        {model.downloaded && !showDownloading ? (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <span>Ready</span>
            </div>
            <Button
              size="sm"
              onClick={onDelete}
              variant="outline"
              disabled={model.loaded}
              title={model.loaded ? 'Unload model before deleting' : 'Delete model'}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ) : showDownloading ? (
          <Button size="sm" variant="outline" disabled>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Downloading...
          </Button>
        ) : (
          <Button size="sm" onClick={onDownload} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        )}
      </div>
    </div>
  );
}
