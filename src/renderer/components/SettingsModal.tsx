import React, { useState } from 'react';
import './SettingsModal.css';

export type SettingsTab = 'general' | 'models' | 'providers' | 'about';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave?: () => void;
  initialTab?: SettingsTab;
}

/**
 * SettingsModal Component - Clinical Zen Settings Panel
 * Modal overlay with tabbed navigation for app settings
 * Requirements: 14.1-14.4, 14.8, 14.9, 14.10
 */
export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  onSave,
  initialTab = 'general',
}) => {
  const [activeTab, setActiveTab] = useState<SettingsTab>(initialTab);
  const [isSaving, setIsSaving] = useState(false);

  if (!isOpen) return null;

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onSave?.();
      onClose();
    } finally {
      setIsSaving(false);
    }
  };

  const tabs: { id: SettingsTab; label: string }[] = [
    { id: 'general', label: 'General' },
    { id: 'models', label: 'Models' },
    { id: 'providers', label: 'Providers' },
    { id: 'about', label: 'About' },
  ];

  return (
    <div className="settings-modal__backdrop" onClick={handleBackdropClick}>
      <div className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <div className="settings-modal__header">
          <h2 id="settings-title" className="settings-modal__title">Settings</h2>
          <button
            className="settings-modal__close-btn"
            onClick={onClose}
            aria-label="Close settings"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <nav className="settings-modal__tabs" role="tablist">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`panel-${tab.id}`}
              className={`settings-modal__tab ${activeTab === tab.id ? 'settings-modal__tab--active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <div className="settings-modal__content">
          <div
            role="tabpanel"
            id={`panel-${activeTab}`}
            aria-labelledby={activeTab}
            className="settings-modal__panel"
          >
            {activeTab === 'general' && <GeneralTab />}
            {activeTab === 'models' && <ModelsTab />}
            {activeTab === 'providers' && <ProvidersTab />}
            {activeTab === 'about' && <AboutTab />}
          </div>
        </div>

        <div className="settings-modal__footer">
          <button
            className="settings-modal__btn settings-modal__btn--cancel"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="settings-modal__btn settings-modal__btn--save"
            onClick={handleSave}
            disabled={isSaving}
          >
            {isSaving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
};

// Placeholder tab components - to be implemented in future tasks
const GeneralTab: React.FC = () => (
  <div className="settings-tab">
    <h3 className="settings-tab__heading">General Settings</h3>
    <p className="settings-tab__description">Configure general application preferences.</p>
  </div>
);

const ModelsTab: React.FC = () => {
  const [geminiApiKey, setGeminiApiKey] = useState('');
  const [openaiApiKey, setOpenaiApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini');
  const [selectedProvider, setSelectedProvider] = useState<'openai' | 'gemini'>('openai');
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [currentStatus, setCurrentStatus] = useState<{ initialized: boolean; provider: string; currentModel: string } | null>(null);

  // Load current API keys and status on mount
  React.useEffect(() => {
    const loadApiKeys = async () => {
      try {
        // Load from localStorage
        const keys = localStorage.getItem('api_keys');
        if (keys) {
          const parsed = JSON.parse(keys);
          setGeminiApiKey(parsed.gemini || '');
          setOpenaiApiKey(parsed.openai || '');
          setSelectedProvider(parsed.provider || 'openai');
        }
        
        const model = localStorage.getItem('selected_model');
        if (model) {
          setSelectedModel(model);
        }

        // Get current AI service status
        const status = await window.api.api.getStatus();
        setCurrentStatus(status);
      } catch (error) {
        console.error('[ModelsTab] Failed to load API keys:', error);
      }
    };
    loadApiKeys();
  }, []);

  // Update model options when provider changes
  const handleProviderChange = (provider: 'openai' | 'gemini') => {
    setSelectedProvider(provider);
    // Set default model for the provider
    if (provider === 'openai') {
      setSelectedModel('gpt-4o-mini');
    } else {
      setSelectedModel('gemini-2.5-flash-lite');
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setSaveStatus('idle');
    
    try {
      // Validate that the selected provider has an API key
      if (selectedProvider === 'openai' && !openaiApiKey) {
        throw new Error('OpenAI API key is required');
      }
      if (selectedProvider === 'gemini' && !geminiApiKey) {
        throw new Error('Gemini API key is required');
      }

      // Save to localStorage
      localStorage.setItem('api_keys', JSON.stringify({
        gemini: geminiApiKey,
        openai: openaiApiKey,
        provider: selectedProvider,
      }));
      localStorage.setItem('selected_model', selectedModel);
      
      // Update API keys in main process
      const result = await window.api.api.setKeys({
        openaiApiKey: openaiApiKey || undefined,
        geminiApiKey: geminiApiKey || undefined,
        selectedModel,
        provider: selectedProvider,
      });
      
      if (!result.success) {
        throw new Error(result.error || 'Failed to set API keys');
      }
      
      // Refresh status
      const status = await window.api.api.getStatus();
      setCurrentStatus(status);
      
      setSaveStatus('success');
      
      // Show success message briefly
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch (error) {
      console.error('[ModelsTab] Failed to save API keys:', error);
      setSaveStatus('error');
    } finally {
      setIsSaving(false);
    }
  };

  const maskApiKey = (key: string) => {
    if (!key || key.length < 8) return key;
    return key.slice(0, 4) + '•'.repeat(key.length - 8) + key.slice(-4);
  };

  const getProviderModels = () => {
    if (selectedProvider === 'openai') {
      return [
        { value: 'gpt-4o', label: 'GPT-4o (Best)' },
        { value: 'gpt-4o-mini', label: 'GPT-4o Mini (Fast & Cheap)' },
        { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
      ];
    } else {
      return [
        { value: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite (Fast)' },
        { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro (Best)' },
      ];
    }
  };

  return (
    <div className="settings-tab">
      <h3 className="settings-tab__heading">AI Models</h3>
      <p className="settings-tab__description">Configure AI providers and model settings.</p>
      
      {currentStatus && (
        <div className="settings-tab__status-banner" style={{
          padding: 'var(--spacing-3)',
          marginBottom: 'var(--spacing-4)',
          backgroundColor: currentStatus.initialized ? 'var(--color-sage-50)' : '#fef3c7',
          borderRadius: 'var(--radius-md)',
          fontSize: 'var(--font-size-small)',
        }}>
          <strong>Status:</strong> {currentStatus.initialized 
            ? `✓ AI Service active (${currentStatus.provider} - ${currentStatus.currentModel})` 
            : '⚠ No API key configured - Please configure below'}
        </div>
      )}
      
      <div className="settings-tab__section">
        <h4 className="settings-tab__section-title">Provider Selection</h4>
        <p className="settings-tab__section-description">
          Choose your AI provider. OpenAI is recommended for best results.
        </p>
        
        <div className="settings-tab__form-group">
          <label className="settings-tab__label">Active Provider</label>
          <div style={{ display: 'flex', gap: 'var(--spacing-3)' }}>
            <button
              type="button"
              onClick={() => handleProviderChange('openai')}
              style={{
                flex: 1,
                padding: 'var(--spacing-3)',
                border: selectedProvider === 'openai' ? '2px solid var(--color-sage-500)' : '1px solid var(--color-gray-200)',
                borderRadius: 'var(--radius-md)',
                backgroundColor: selectedProvider === 'openai' ? 'var(--color-sage-50)' : 'transparent',
                cursor: 'pointer',
              }}
            >
              <strong>OpenAI</strong>
              <div style={{ fontSize: 'var(--font-size-tiny)', color: 'var(--color-text-secondary)' }}>
                GPT-4o, GPT-4o Mini
              </div>
            </button>
            <button
              type="button"
              onClick={() => handleProviderChange('gemini')}
              style={{
                flex: 1,
                padding: 'var(--spacing-3)',
                border: selectedProvider === 'gemini' ? '2px solid var(--color-sage-500)' : '1px solid var(--color-gray-200)',
                borderRadius: 'var(--radius-md)',
                backgroundColor: selectedProvider === 'gemini' ? 'var(--color-sage-50)' : 'transparent',
                cursor: 'pointer',
              }}
            >
              <strong>Google Gemini</strong>
              <div style={{ fontSize: 'var(--font-size-tiny)', color: 'var(--color-text-secondary)' }}>
                Gemini 2.5 Flash, Pro
              </div>
            </button>
          </div>
        </div>
      </div>

      <div className="settings-tab__section">
        <h4 className="settings-tab__section-title">API Keys</h4>
        <p className="settings-tab__section-description">
          Enter your API key for the selected provider. Keys are stored locally and never sent to external servers.
        </p>
        
        {selectedProvider === 'openai' && (
          <div className="settings-tab__form-group">
            <label htmlFor="openai-api-key" className="settings-tab__label">
              OpenAI API Key *
            </label>
            <input
              id="openai-api-key"
              type="password"
              className="settings-tab__input"
              placeholder="sk-..."
              value={openaiApiKey}
              onChange={(e) => setOpenaiApiKey(e.target.value)}
            />
            <p className="settings-tab__hint">
              Get your API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer">OpenAI Platform</a>
            </p>
          </div>
        )}

        {selectedProvider === 'gemini' && (
          <div className="settings-tab__form-group">
            <label htmlFor="gemini-api-key" className="settings-tab__label">
              Google Gemini API Key *
            </label>
            <input
              id="gemini-api-key"
              type="password"
              className="settings-tab__input"
              placeholder="AIza..."
              value={geminiApiKey}
              onChange={(e) => setGeminiApiKey(e.target.value)}
            />
            <p className="settings-tab__hint">
              Get your API key from <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a>
            </p>
          </div>
        )}
      </div>

      <div className="settings-tab__section">
        <h4 className="settings-tab__section-title">Model Selection</h4>
        <p className="settings-tab__section-description">
          Choose which model to use for chat responses.
        </p>
        
        <div className="settings-tab__form-group">
          <label htmlFor="model-select" className="settings-tab__label">
            Active Model
          </label>
          <select
            id="model-select"
            className="settings-tab__select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {getProviderModels().map(model => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="settings-tab__actions">
        <button
          className="settings-tab__save-btn"
          onClick={handleSave}
          disabled={isSaving}
        >
          {isSaving ? 'Saving...' : 'Save Configuration'}
        </button>
        
        {saveStatus === 'success' && (
          <span className="settings-tab__status settings-tab__status--success">
            ✓ Configuration saved successfully
          </span>
        )}
        
        {saveStatus === 'error' && (
          <span className="settings-tab__status settings-tab__status--error">
            ✗ Failed to save configuration
          </span>
        )}
      </div>

      {(openaiApiKey || geminiApiKey) && (
        <div className="settings-tab__info-box">
          <h4 className="settings-tab__info-title">Current Configuration</h4>
          <p className="settings-tab__info-text">
            <strong>Provider:</strong> {selectedProvider === 'openai' ? 'OpenAI' : 'Google Gemini'}
          </p>
          {openaiApiKey && (
            <p className="settings-tab__info-text">
              <strong>OpenAI API Key:</strong> {maskApiKey(openaiApiKey)}
            </p>
          )}
          {geminiApiKey && (
            <p className="settings-tab__info-text">
              <strong>Gemini API Key:</strong> {maskApiKey(geminiApiKey)}
            </p>
          )}
          <p className="settings-tab__info-text">
            <strong>Active Model:</strong> {selectedModel}
          </p>
        </div>
      )}
    </div>
  );
};

const ProvidersTab: React.FC = () => (
  <div className="settings-tab">
    <h3 className="settings-tab__heading">Providers</h3>
    <p className="settings-tab__description">Configure AI providers and API connections.</p>
  </div>
);

const AboutTab: React.FC = () => (
  <div className="settings-tab">
    <h3 className="settings-tab__heading">About GoatedApp</h3>
    <p className="settings-tab__description">
      A privacy-first clinical orchestration platform.
    </p>
    <div className="settings-tab__info">
      <p>Version: 1.0.0</p>
      <p>All patient data remains on-device.</p>
    </div>
  </div>
);

export default SettingsModal;
