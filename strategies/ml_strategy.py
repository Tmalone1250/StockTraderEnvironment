"""Machine Learning Enhanced Trading Strategy"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from .base import State

class MLState(State):
    """Extended state class to handle ML model state"""
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.training_window = 60  # Days of data to train on
        self.prediction_threshold = 0.001  # 0.1% price movement threshold
        self.feature_means = None  # Store feature means for NaN handling
        self.last_predictions = []  # Store recent predictions for momentum
        
    def calculate_momentum_features(self, df, current_step):
        """Calculate additional momentum-based features"""
        try:
            # Price momentum
            returns_5d = (df['Close'].iloc[current_step] / df['Close'].iloc[current_step - 5]) - 1
            returns_10d = (df['Close'].iloc[current_step] / df['Close'].iloc[current_step - 10]) - 1
            returns_20d = (df['Close'].iloc[current_step] / df['Close'].iloc[current_step - 20]) - 1
            
            # Volume momentum
            vol_ratio_5d = df['Volume'].iloc[current_step] / df['Volume'].iloc[current_step-5:current_step].mean()
            vol_ratio_10d = df['Volume'].iloc[current_step] / df['Volume'].iloc[current_step-10:current_step].mean()
            
            # Trend strength
            ma_ratios = [
                df['Close'].iloc[current_step] / df['MA10'].iloc[current_step],
                df['MA10'].iloc[current_step] / df['MA30'].iloc[current_step]
            ]
            trend_strength = np.mean([abs(ratio - 1) for ratio in ma_ratios])
            
            # Volatility features
            recent_volatility = df['Returns'].iloc[current_step-10:current_step].std() * np.sqrt(252)
            volatility_ratio = recent_volatility / df['Volatility'].iloc[current_step]
            
            return [
                returns_5d,
                returns_10d,
                returns_20d,
                vol_ratio_5d,
                vol_ratio_10d,
                trend_strength,
                volatility_ratio
            ]
        except Exception as e:
            print(f"Error calculating momentum features: {str(e)}")
            return [0] * 7
        
    def prepare_features(self, df, current_step):
        """Prepare feature set for ML model"""
        if current_step < self.training_window:
            return None
            
        # Technical indicators as features
        features = []
        try:
            # Use last 10 days of data
            for i in range(current_step - 10, current_step):
                # Base technical indicators
                row = [
                    df['BB_PCT'].iloc[i],
                    df['BB_Width'].iloc[i],
                    df['RSI'].iloc[i] / 100,  # Normalize RSI
                    df['ROC_MA'].iloc[i] / 100,  # Normalize ROC
                    df['MACD'].iloc[i],
                    df['Signal'].iloc[i],
                    df['Trend_Strength'].iloc[i],
                    df['Strong_Trend'].iloc[i],
                    df['Volatility'].iloc[i],
                    df['ATR'].iloc[i] / df['Close'].iloc[i],  # Normalize ATR
                    1 if df['MA10'].iloc[i] > df['MA30'].iloc[i] else 0,  # Trend direction
                    df['Price_Slope'].iloc[i] / 100,  # Normalize slope
                ]
                
                # Add momentum features
                momentum_features = self.calculate_momentum_features(df, i)
                row.extend(momentum_features)
                
                features.extend(row)
                
            features = np.array(features)
            
            # Handle NaN values using stored means
            if self.feature_means is not None:
                features = np.nan_to_num(features, nan=self.feature_means)
                
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return None
        
    def calculate_position_size(self, predicted_return, df, current_step):
        """Calculate position size based on multiple factors"""
        try:
            base_position = 1.0
            
            # Prediction confidence
            confidence_scale = min(abs(predicted_return) / self.prediction_threshold, 2.5)
            base_position *= confidence_scale
            
            # Trend alignment
            trend_aligned = (
                predicted_return > 0 and df['MA10'].iloc[current_step] > df['MA30'].iloc[current_step] or
                predicted_return < 0 and df['MA10'].iloc[current_step] < df['MA30'].iloc[current_step]
            )
            if trend_aligned:
                base_position *= 1.3
                
            # Momentum confirmation
            momentum_confirmed = (
                predicted_return > 0 and df['ROC_MA'].iloc[current_step] > 0 or
                predicted_return < 0 and df['ROC_MA'].iloc[current_step] < 0
            )
            if momentum_confirmed:
                base_position *= 1.2
                
            # Volume confirmation
            volume_ratio = df['Volume'].iloc[current_step] / df['Volume'].iloc[current_step-10:current_step].mean()
            if volume_ratio > 1.2:
                base_position *= 1.1
                
            # Volatility adjustment
            volatility = df['Volatility'].iloc[current_step]
            volatility_factor = 1.0 / (1.0 + volatility)
            base_position *= volatility_factor
            
            # Market regime adjustment
            if df['Strong_Trend'].iloc[current_step]:
                base_position *= 1.2
                
            # Recent performance adjustment
            if len(self.last_predictions) >= 5:
                accuracy = np.mean([1 if np.sign(pred) == np.sign(actual) else 0 
                                  for pred, actual in self.last_predictions[-5:]])
                base_position *= (0.8 + 0.4 * accuracy)  # Scale between 0.8 and 1.2
                
            return min(base_position, 2.0)  # Cap at 200%
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return 1.0
        
    def update_prediction_history(self, predicted_return, actual_return):
        """Update prediction history for tracking accuracy"""
        self.last_predictions.append((predicted_return, actual_return))
        if len(self.last_predictions) > 10:
            self.last_predictions.pop(0)
            
    def prepare_training_data(self, df, current_step):
        """Prepare training data for ML model"""
        if current_step < self.training_window:
            return None, None
            
        X, y = [], []
        
        try:
            # Create training samples from historical data
            for i in range(self.training_window, current_step):
                features = []
                for j in range(i - 10, i):
                    # Base technical indicators
                    row = [
                        df['BB_PCT'].iloc[j],
                        df['BB_Width'].iloc[j],
                        df['RSI'].iloc[j] / 100,
                        df['ROC_MA'].iloc[j] / 100,
                        df['MACD'].iloc[j],
                        df['Signal'].iloc[j],
                        df['Trend_Strength'].iloc[j],
                        df['Strong_Trend'].iloc[j],
                        df['Volatility'].iloc[j],
                        df['ATR'].iloc[j] / df['Close'].iloc[j],
                        1 if df['MA10'].iloc[j] > df['MA30'].iloc[j] else 0,
                        df['Price_Slope'].iloc[j] / 100,
                    ]
                    
                    # Add momentum features
                    momentum_features = self.calculate_momentum_features(df, j)
                    row.extend(momentum_features)
                    
                    features.extend(row)
                
                # Only add samples with valid data
                if not any(np.isnan(features)):
                    X.append(features)
                    # Target: 5-day forward returns
                    future_return = (df['Close'].iloc[min(i + 5, current_step)] / df['Close'].iloc[i]) - 1
                    y.append(future_return)
                
            if len(X) > 0:
                X = np.array(X)
                y = np.array(y)
                
                # Store feature means for NaN handling
                self.feature_means = np.nanmean(X, axis=0)
                
                # Handle any remaining NaN values
                X = np.nan_to_num(X, nan=self.feature_means)
                y = np.nan_to_num(y, nan=0)
                
                return X, y
                
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            
        return None, None
        
    def train_model(self, X, y):
        """Train the ML model"""
        if X is None or len(X) < 10:  # Need enough samples
            return
            
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize and train model
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),  # Deeper network
                activation='relu',
                solver='adam',
                alpha=0.005,  # Moderate regularization
                batch_size=32,
                learning_rate='adaptive',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=10,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.trained = True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
        
    def predict_returns(self, features):
        """Predict future returns using the trained model"""
        if not self.trained or features is None:
            return 0
            
        try:
            features_scaled = self.scaler.transform(features)
            return self.model.predict(features_scaled)[0]
        except Exception as e:
            print(f"Error predicting returns: {str(e)}")
            return 0

def ml_enhanced_strategy(state, df, current_step):
    """
    Machine Learning Enhanced Trading Strategy
    - Uses neural network to predict short-term price movements
    - Enhanced feature engineering with momentum indicators
    - Sophisticated position sizing based on multiple factors
    - Adaptive to market regimes and volatility
    """
    try:
        if not isinstance(state, MLState):
            state = MLState()
            
        if current_step < 60:  # Need enough data for training
            return 0.0, state
            
        # Get current indicators
        close = df['Close'].iloc[current_step]
        volatility = df['Volatility'].iloc[current_step]
        
        # Update risk metrics
        portfolio_value = close * state.position_size if state.position_size else close
        state.update_risk_metrics(close, portfolio_value, current_step)
        
        # Check if we should exit based on risk management
        if state.should_exit_trade(close, volatility):
            return 0.0, state
            
        # Prepare and train model if needed
        if not state.trained and current_step >= state.training_window:
            X, y = state.prepare_training_data(df, current_step)
            if X is not None and y is not None:
                state.train_model(X, y)
            
        # Get prediction for current market state
        features = state.prepare_features(df, current_step)
        predicted_return = state.predict_returns(features)
        
        # Update prediction history if we have actual returns
        if current_step > 5 and len(state.last_predictions) > 0:
            actual_return = (close / df['Close'].iloc[current_step - 5]) - 1
            state.update_prediction_history(state.last_predictions[-1][0], actual_return)
        
        # Calculate position size based on multiple factors
        if abs(predicted_return) > state.prediction_threshold:
            position_size = state.calculate_position_size(predicted_return, df, current_step)
            
            # Determine final position based on prediction direction
            if predicted_return > state.prediction_threshold:
                final_position = position_size
            elif predicted_return < -state.prediction_threshold:
                final_position = 0.0
            else:
                final_position = state.position_size if state.position_size else 0.0
        else:
            final_position = 0.0  # No strong signal
            
        # Ensure position size doesn't exceed max
        final_position = min(final_position, 1.0)
        
        # Update state
        state.position_size = final_position
        state.entry_price = close
        
        return final_position, state
        
    except Exception as e:
        print(f"Error in ML strategy: {str(e)}")
        return 0.0, state
