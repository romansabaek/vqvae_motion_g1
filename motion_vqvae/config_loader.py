"""
Configuration loader for MotionVQVAE
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and manage configuration from YAML files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract agent config
        if 'agent' in config and 'config' in config['agent']:
            self.config = config['agent']['config']
        else:
            self.config = config
        
        # Ensure proper type conversion for numeric values
        self._convert_numeric_values()
        
        # Validate required parameters
        self._validate_config()
        
        logger.info(f"Loaded configuration from: {config_path}")
        return self.config
    
    def _convert_numeric_values(self):
        """Convert string values to appropriate numeric types."""
        numeric_keys = [
            'lr', 'weight_decay', 'gamma', 'commit', 'loss_vel', 'beta', 'mu',
            'total_iter', 'warmup_iter', 'print_iter', 'save_every', 'seed',
            'batch_size', 'window_size', 'num_workers', 'num_joints', 'frame_size',
            'code_dim', 'nb_code', 'down_t', 'stride_t', 'width', 'depth',
            'dilation_growth_rate', 'output_emb_width'
        ]
        
        for key in numeric_keys:
            if key in self.config:
                value = self.config[key]
                if isinstance(value, str):
                    try:
                        # Try to convert to float first
                        if '.' in value or 'e' in value.lower():
                            self.config[key] = float(value)
                        else:
                            self.config[key] = int(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
    
    def _validate_config(self):
        """Validate that all required parameters are present."""
        required_params = [
            'batch_size', 'window_size', 'lr', 'total_iter', 'warmup_iter',
            'code_dim', 'nb_code', 'down_t', 'stride_t', 'width', 'depth',
            'dilation_growth_rate', 'output_emb_width', 'vq_act', 'vq_norm',
            'commit', 'loss_vel', 'use_wandb', 'wandb_project', 'wandb_run_name',
            'wandb_tags', 'save_every', 'print_iter'
        ]
        
        missing_params = []
        for param in required_params:
            if param not in self.config:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        self.config.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()
    
    def save_config(self, save_path: str):
        """Save current configuration to YAML file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to: {save_path}")


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    """
    loader = ConfigLoader(config_path)
    return loader.to_dict()


def create_config_from_yaml(config_path: str, **overrides) -> Dict[str, Any]:
    """
    Load configuration from YAML file and apply overrides.
    """
    config = load_config_from_yaml(config_path)
    config.update(overrides)
    return config
