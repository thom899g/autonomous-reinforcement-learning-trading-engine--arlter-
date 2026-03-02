"""
ARLTER Configuration Module
Centralized configuration management with environment variable support
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    initial_balance: float = 10000.0
    max_position_size: float = 0.2  # 20% of portfolio per trade
    transaction_fee: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit


@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    learning_rate: float = 0.001
    gamma: float = 0.95  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    replay_buffer_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 100  # Steps between target network updates
    hidden_dim: int = 128


@dataclass
class DataConfig:
    """Data fetching and processing configuration"""
    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    lookback_window: int = 100  # Number of historical candles
    feature_count: int = 14  # Technical indicators + price features
    validation_split: float = 0.2


@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    project_id: str = os.getenv("FIREBASE_PROJECT_ID", "arlter-default")
    credential_path: str = os.getenv("FIREBASE_CREDENTIAL_PATH", "./service-account.json")
    collection_name: str = "trading_agents"
    state_subcollection: str = "agent_states"


@dataclass
class SystemConfig:
    """System-level configuration"""
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    checkpoint_interval: int = 1000  # Steps between checkpoints
    max_consecutive_losses: int = 10
    min_training_samples: int = 1000
    telemetry_enabled: bool = True


# Global configuration instances
TRADING_CONFIG = TradingConfig()
RL_CONFIG = RLConfig()
DATA_CONFIG = DataConfig()
FIREBASE_CONFIG = FirebaseConfig()
SYSTEM_CONFIG = SystemConfig()


def validate_config() -> None:
    """Validate all configuration values for consistency"""
    errors = []
    
    if TRADING_CONFIG.initial_balance <= 0:
        errors.append("Initial balance must be positive")
    if not 0 < TRADING_CONFIG.max_position_size <= 1:
        errors.append("Max position size must be between 0 and 1")
    if RL_CONFIG.gamma <= 0 or RL_CONFIG.gamma >= 1:
        errors.append("Discount factor must be between 0 and 1")
    if DATA_CONFIG.lookback_window < 10:
        errors.append("Lookback window must be at least 10 periods")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")