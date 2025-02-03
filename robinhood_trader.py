import robin_stocks.robinhood as rh
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyotp
import os
from dotenv import load_dotenv

class RobinhoodTrader:
    def __init__(self, use_mfa=True):
        """Initialize Robinhood trader with authentication"""
        load_dotenv()  # Load environment variables
        self.username = os.getenv('ROBINHOOD_USERNAME')
        self.password = os.getenv('ROBINHOOD_PASSWORD')
        self.mfa_code = os.getenv('ROBINHOOD_MFA_CODE')
        self.use_mfa = use_mfa
        self.logged_in = False
        
    def login(self):
        """Login to Robinhood account"""
        try:
            if self.use_mfa and self.mfa_code:
                totp = pyotp.TOTP(self.mfa_code).now()
                login = rh.login(self.username, self.password, mfa_code=totp)
            else:
                login = rh.login(self.username, self.password)
            self.logged_in = True
            return True
        except Exception as e:
            print(f"Login failed: {str(e)}")
            return False
    
    def logout(self):
        """Logout from Robinhood account"""
        if self.logged_in:
            rh.logout()
            self.logged_in = False
    
    def get_account_info(self):
        """Get account information"""
        if not self.logged_in:
            return None
        return {
            'portfolio_value': float(rh.load_portfolio_profile()['equity']),
            'buying_power': float(rh.load_account_profile()['buying_power']),
            'cash': float(rh.load_account_profile()['cash']),
        }
    
    def get_positions(self):
        """Get current positions"""
        if not self.logged_in:
            return None
        positions = rh.get_open_stock_positions()
        formatted_positions = []
        for position in positions:
            symbol = rh.get_symbol_by_url(position['instrument'])
            formatted_positions.append({
                'symbol': symbol,
                'quantity': float(position['quantity']),
                'average_price': float(position['average_buy_price']),
                'current_price': float(rh.get_latest_price(symbol)[0]),
            })
        return formatted_positions
    
    def get_stock_data(self, symbol, interval='day', span='year'):
        """Get historical stock data"""
        if not self.logged_in:
            return None
        try:
            historicals = rh.get_stock_historicals(symbol, interval=interval, span=span)
            df = pd.DataFrame(historicals)
            df['begins_at'] = pd.to_datetime(df['begins_at'])
            df.set_index('begins_at', inplace=True)
            for col in ['open_price', 'close_price', 'high_price', 'low_price']:
                df[col] = pd.to_numeric(df[col])
            return df
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return None
    
    def place_order(self, symbol, quantity, side='buy', order_type='market', limit_price=None):
        """Place a stock order"""
        if not self.logged_in:
            return None
        try:
            if order_type == 'market':
                if side == 'buy':
                    order = rh.order_buy_market(symbol, quantity)
                else:
                    order = rh.order_sell_market(symbol, quantity)
            else:
                if side == 'buy':
                    order = rh.order_buy_limit(symbol, quantity, limit_price)
                else:
                    order = rh.order_sell_limit(symbol, quantity, limit_price)
            return order
        except Exception as e:
            print(f"Order failed: {str(e)}")
            return None
    
    def get_order_status(self, order_id):
        """Get status of an order"""
        if not self.logged_in:
            return None
        return rh.get_stock_order_info(order_id)
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        if not self.logged_in:
            return None
        return rh.cancel_stock_order(order_id)
    
    def get_option_positions(self):
        """Get current option positions"""
        if not self.logged_in:
            return None
        return rh.get_open_option_positions()
    
    def get_margin_info(self):
        """Get margin account information"""
        if not self.logged_in:
            return None
        return rh.load_margin_profile()
    
    def place_option_order(self, option_id, quantity, side='buy', order_type='market', limit_price=None):
        """Place an option order"""
        if not self.logged_in:
            return None
        try:
            if order_type == 'market':
                if side == 'buy':
                    order = rh.order_buy_option_market(option_id, quantity)
                else:
                    order = rh.order_sell_option_market(option_id, quantity)
            else:
                if side == 'buy':
                    order = rh.order_buy_option_limit(option_id, quantity, limit_price)
                else:
                    order = rh.order_sell_option_limit(option_id, quantity, limit_price)
            return order
        except Exception as e:
            print(f"Option order failed: {str(e)}")
            return None
