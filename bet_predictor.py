import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import requests
from datetime import datetime, timedelta
import csv
import json
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ValueBetFinder:
    def __init__(self, root):
        self.root = root
        self.root.title("Value Bet Finder")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # API Configuration
        self.BASE_URL = "https://api.sportmonks.com"
        self.MARKETS = {
            "match_odds": [1, 2, 3],  # Home, Draw, Away
            "over_under": [19, 20, 21, 22],  # Over/Under 1.5, 2.5, 3.5
            "btts": [10],  # Both teams to score
            "correct_score": range(40, 66)  # Correct Score markets
        }
        
        # Market mappings for predictions
        self.MARKET_MAPPINGS = {
            'match_odds': {
                'Home': 'home_win_probability',
                'Draw': 'draw_probability',
                'Away': 'away_win_probability'
            },
            'over_under': {
                'Over 1.5': 'over_1_5_probability',
                'Under 1.5': 'under_1_5_probability',
                'Over 2.5': 'over_2_5_probability',
                'Under 2.5': 'under_2_5_probability',
                'Over 3.5': 'over_3_5_probability',
                'Under 3.5': 'under_3_5_probability'
            },
            'btts': {
                'Yes': 'btts_probability',
                'No': 'btts_no_probability'
            }
        }
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('Header.TLabel', font=('Arial', 9, 'bold'))
        self.style.configure('Custom.TEntry', padding=2)
        self.style.configure('Custom.TCombobox', padding=2)
        
        # Create main frame with padding
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Input fields frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create input fields with proper spacing and alignment
        self.create_labeled_input(input_frame, "SPORTMONKS PLAN", combobox=True, 
                                values=['EUROPEAN', 'WORLDWIDE', 'CUSTOM'], 
                                default='EUROPEAN', row=0)
        self.create_labeled_input(input_frame, "API TOKEN", 
                                default="zlm2XyjwUTq2nKk1QJM0TB4t0hP0j5RepYyyz9HdbZUEXcPKLaTtHjjfhzf9", 
                                row=1)
        self.create_labeled_input(input_frame, "TIME INTERVAL", combobox=True,
                                values=['4 HRS', '6 HRS', '8 HRS', '12 HRS', '16 HRS', '24 HRS'],
                                default='6 HRS', row=2)
        self.create_labeled_input(input_frame, "VALUE BET THRESHOLD", default="11.57", row=3)
        self.create_labeled_input(input_frame, "BETFAIR COMMISSION", default="6.52", row=4)
        self.create_labeled_input(input_frame, "BETFAIR DISCOUNT", default="20", row=5)
        self.create_labeled_input(input_frame, "BET UNIT", default="8", row=6)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="GET VALUE BET LIST", 
                  command=self.get_value_bets).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="EXPORT TO CSV",
                  command=self.export_to_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="REFRESH LIST",
                  command=self.refresh_list).pack(side=tk.LEFT, padx=2)
        
        # Create Treeview
        self.create_treeview(main_frame)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(root, textvariable=self.status_var, 
                                  relief=tk.SUNKEN, anchor=tk.W, wraplength=780)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status('Application initialized successfully.')

    def create_labeled_input(self, parent, label, combobox=False, values=None, default="", row=0):
        """Create labeled input field"""
        label_widget = ttk.Label(parent, text=label, style='Header.TLabel')
        label_widget.grid(row=row, column=0, sticky=tk.W, pady=2)
        
        if combobox:
            var = tk.StringVar(value=default)
            widget = ttk.Combobox(parent, textvariable=var, values=values, 
                                width=30, style='Custom.TCombobox')
            widget.set(default)
        else:
            var = tk.StringVar(value=default)
            widget = ttk.Entry(parent, textvariable=var, width=32, 
                             style='Custom.TEntry')
        
        widget.grid(row=row, column=1, sticky=tk.W, pady=2)
        var_name = label.lower().replace(' ', '_').replace('%', '').replace('-', '_')
        setattr(self, var_name, var)
        return var

    def create_treeview(self, parent):
        """Create and configure the Treeview"""
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('event', 'betfair_adjust', 'spmk_odds', 'value', 'spmk_stake', 'money_to_bet')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        headings = ('EVENT', 'BETFAIR ADJUST', 'SPMK ODDS', 'VALUE', 'SPMK STAKE', 'MONEY TO BET')
        for col, heading in zip(columns, headings):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=100)
        
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        
        self.tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        y_scrollbar.grid(row=0, column=1, sticky='ns')
        x_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

    def update_status(self, message, level="INFO"):
        """Update status bar with message"""
        timestamp = datetime.now().strftime("%H:%M:%S %p")
        self.status_var.set(f'["{level}" - {timestamp}] {message}')
        logger.info(message)

    def get_fixture_dates(self):
        """Get start and end dates based on time interval"""
        hours = int(self.time_interval.get().split()[0])
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)
        return start_date, end_date

    def get_fixtures(self, headers):
        """Get fixtures for the specified date range"""
        start_date, end_date = self.get_fixture_dates()
        fixtures_url = f"{self.BASE_URL}/v3/football/fixtures/between/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        self.update_status(f"Fetching fixtures list...")
        response = requests.get(fixtures_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        return data.get('data', [])

    def get_fixtures_detail(self, fixture_ids, headers):
        """Get detailed fixture information using multi endpoint"""
        if not fixture_ids:
            return []
            
        multi_url = f"{self.BASE_URL}/v3/football/fixtures/multi/{','.join(map(str, fixture_ids))}"
        
        self.update_status(f"Fetching detailed fixture data...")
        response = requests.get(multi_url, headers=headers)
        response.raise_for_status()
        
        return response.json().get('data', [])

    def get_fixture_predictions(self, fixture_id, headers):
        """Get predictions for a fixture"""
        url = f"{self.BASE_URL}/v3/football/predictions/probabilities/fixtures/{fixture_id}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get('data', {})
        return {}

    def get_fixture_odds(self, fixture_id, headers):
        """Get odds for a fixture"""
        url = f"{self.BASE_URL}/v3/odds/bookmakers/fixtures/{fixture_id}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json().get('data', [])
            return next((bm for bm in data if bm.get('name', '').lower() == 'betfair'), None)
        return None

    def get_sportmonks_data(self):
        """Main method to fetch all required data"""
        api_token = self.api_token.get()
        headers = {"Authorization": f"{api_token}"}
        
        try:
            # Get initial fixtures list
            fixtures = self.get_fixtures(headers)
            if not fixtures:
                self.update_status("No fixtures found for the specified period", "WARNING")
                return []
            
            # Process fixtures in batches
            batch_size = 10
            combined_data = []
            
            for i in range(0, len(fixtures), batch_size):
                batch = fixtures[i:i + batch_size]
                batch_ids = [fixture['id'] for fixture in batch]
                
                self.update_status(f"Processing batch {i//batch_size + 1} of {(len(fixtures)-1)//batch_size + 1}...")
                
                # Get detailed fixture information
                detailed_fixtures = self.get_fixtures_detail(batch_ids, headers)
                
                # Get predictions and odds for each fixture
                for fixture in detailed_fixtures:
                    fixture_id = fixture['id']
                    
                    # Get predictions
                    predictions = self.get_fixture_predictions(fixture_id, headers)
                    if predictions:
                        fixture['predictions'] = predictions
                    
                    # Get Betfair odds
                    betfair_odds = self.get_fixture_odds(fixture_id, headers)
                    if betfair_odds:
                        fixture['betfair_odds'] = betfair_odds
                    
                    if 'predictions' in fixture and 'betfair_odds' in fixture:
                        combined_data.append(fixture)
                    
                    time.sleep(0.2)  # Rate limiting
            
            self.update_status(f"Retrieved complete data for {len(combined_data)} fixtures")
            return combined_data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API Error: {str(e)}"
            self.update_status(error_msg, "ERROR")
            messagebox.showerror("API Error", error_msg)
            return []

    def calculate_betfair_adjust(self, odds):
        """Calculate adjusted Betfair odds considering commission and discount"""
        commission = float(self.betfair_commission.get()) / 100
        discount = float(self.betfair_discount.get()) / 100
        return round(1 + (odds - 1) * (1 - commission * (1 - discount)), 2)

    def get_prediction_key(self, market_type, selection_name):
        """Get the corresponding prediction key for a market selection"""
        try:
            if market_type == 'btts' and selection_name == 'No':
                # Special handling for BTTS No
                return lambda x: 100 - float(x.get('btts_probability', 0))
            return self.MARKET_MAPPINGS.get(market_type, {}).get(selection_name)
        except:
            return None

    def get_value_bets(self):
        """Main method to find and display value bets"""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Get data
            fixtures_data = self.get_sportmonks_data()
            if not fixtures_data:
                return
            
            value_threshold = float(self.value_bet_threshold.get())
            matches_processed = 0
            value_bets_found = 0
            
            for fixture in fixtures_data:
                matches_processed += 1
                
                predictions = fixture.get('predictions', {})
                betfair_odds = fixture.get('betfair_odds', {})
                
                for market_type, market_ids in self.MARKETS.items():
                    for market in betfair_odds.get('markets', []):
                        market_id = market.get('market_id')
                        
                        if market_id not in market_ids:
                            continue
                        
                        try:
                            for selection in market.get('selections', []):
                                betfair_price = float(selection.get('odds', 0))
                                if betfair_price <= 0:
                                    continue
                                
                                # Get prediction probability
                                pred_key = self.get_prediction_key(market_type, selection['name'])
                                if callable(pred_key):
                                    probability = pred_key(predictions) / 100
                                elif pred_key in predictions:
                                    probability = float(predictions[pred_key]) / 100
                                else:
                                    continue
                                
                                if probability <= 0:
                                    continue
                                
                                true_odds = 1 / probability
                                betfair_adjust = self.calculate_betfair_adjust(betfair_price)
                                value = ((betfair_adjust / true_odds) - 1) * 100
                                
                                if abs(value) >= value_threshold:
                                    event_name = (f"{fixture.get('home_team', {}).get('name', 'Unknown')} vs "
                                                f"{fixture.get('away_team', {}).get('name', 'Unknown')}")
                                    market_name = f"{market.get('name', 'Unknown Market')} - {selection.get('name', 'Unknown Selection')}"
                                    full_event = f"{event_name} ({market_name})"
                                    
                                    money_to_bet = float(self.bet_unit.get()) * true_odds
                                    
                                    tag = 'positive' if value > 0 else 'negative'
                                    self.tree.insert('', tk.END, values=(
                                        full_event,
                                        f"{betfair_adjust:.2f}",
                                        f"{true_odds:.2f}",
                                        f"{value:.2f}%",
                                        f"{true_odds:.2f}",
                                        f"{money_to_bet:.2f}"
                                    ), tags=(tag,))
                                    value_bets_found += 1
                        
                        except (ValueError, KeyError, TypeError) as e:
                            logger.error(f"Error processing market: {str(e)}")
                            continue
            
            # Configure colors for positive and negative values
            self.tree.tag_configure('positive', background='#90EE90')  # Light green
            self.tree.tag_configure('negative', background='#FFB6C1')  # Light red
            
            self.update_status(
                f"Processed {matches_processed} matches, found {value_bets_found} value bets"
            )
            
        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            self.update_status(error_msg, "ERROR")
            messagebox.showerror("Processing Error", error_msg)

    def export_to_csv(self):
        """Export value bets to CSV in BF Botmanager format"""
        filename = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[("CSV files", "*.csv")]
        )
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    # Headers according to BF Botmanager format
                    writer.writerow([
                        "Date", "Time", "Event", "Market", "Selection",
                        "Back/Lay", "Odds", "Stake", "Value %", "Sport",
                        "Competition", "Market Type", "In-Play"
                    ])
                    
                    current_time = datetime.now()
                    date_str = current_time.strftime("%Y-%m-%d")
                    time_str = current_time.strftime("%H:%M")
                    
                    for item in self.tree.get_children():
                        values = self.tree.item(item)['values']
                        event_parts = values[0].split(' (')
                        event = event_parts[0]
                        market_info = event_parts[1].rstrip(')')
                        market, selection = market_info.split(' - ', 1)
                        
                        writer.writerow([
                            date_str,           # Date
                            time_str,           # Time
                            event,              # Event
                            market,             # Market
                            selection,          # Selection
                            "Back",             # Back/Lay
                            values[1],          # Odds
                            values[5],          # Stake
                            values[3],          # Value %
                            "Football",         # Sport
                            "",                 # Competition (optional)
                            "Match Odds",       # Market Type
                            "false"             # In-Play
                        ])
                
                self.update_status(f"Successfully exported value bets to {filename}")
            
            except Exception as e:
                error_msg = f"Error exporting to CSV: {str(e)}"
                self.update_status(error_msg, "ERROR")
                messagebox.showerror("Export Error", error_msg)

    def refresh_list(self):
        """Refresh the value bets list"""
        self.get_value_bets()

def main():
    root = tk.Tk()
    app = ValueBetFinder(root)
    root.mainloop()

if __name__ == "__main__":
    main()