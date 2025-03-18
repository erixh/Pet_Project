import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class BinomialModel:
    def __init__(self, S0, K, T, r, sigma, n_steps, option_type='call'):
        """
        Initialize the binomial model for option pricing.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        T : float
            Time to expiration in years
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility of the underlying asset (annualized)
        n_steps : int
            Number of time steps in the binomial tree
        option_type : str
            Type of option ('call' or 'put')
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.option_type = option_type.lower()
        
        # Calculate time step
        self.dt = T / n_steps
        
        # Calculate up and down factors
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        
        # Calculate risk-neutral probability
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        
    def build_price_tree(self):
        """Build the binomial tree for the underlying asset prices."""
        # Initialize price tree
        price_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # Fill the price tree
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                # Number of up moves
                up_moves = j
                # Number of down moves
                down_moves = i - j
                # Calculate price at this node
                price_tree[j, i] = self.S0 * (self.u ** up_moves) * (self.d ** down_moves)
                
        return price_tree
    
    def build_option_tree(self, price_tree):
        """Build the option pricing tree based on the asset price tree."""
        # Initialize option tree
        option_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # Fill the option values at expiration (last column)
        for j in range(self.n_steps + 1):
            if self.option_type == 'call':
                option_tree[j, self.n_steps] = max(0, price_tree[j, self.n_steps] - self.K)
            else:  # put option
                option_tree[j, self.n_steps] = max(0, self.K - price_tree[j, self.n_steps])
        
        # Backward induction to fill the option tree
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Calculate option value at this node (discounted expected value)
                option_tree[j, i] = np.exp(-self.r * self.dt) * (
                    self.p * option_tree[j + 1, i + 1] + 
                    (1 - self.p) * option_tree[j, i + 1]
                )
                
        return option_tree
    
    def price_option(self):
        """Calculate the option price using the binomial model."""
        price_tree = self.build_price_tree()
        option_tree = self.build_option_tree(price_tree)
        
        # Option price is at the root of the option tree
        return option_tree[0, 0]
    
    def plot_trees(self):
        """Plot both the price tree and option value tree."""
        price_tree = self.build_price_tree()
        option_tree = self.build_option_tree(price_tree)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot price tree
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                # Plot node
                ax1.scatter(i, j, color='blue')
                # Add price label
                ax1.annotate(f"{price_tree[j, i]:.2f}", (i, j), 
                           textcoords="offset points", xytext=(0, 5), 
                           ha='center')
                
                # Draw lines to children if not at the last step
                if i < self.n_steps:
                    # Line to upper child
                    ax1.plot([i, i+1], [j, j+1], 'k-', alpha=0.3)
                    # Line to lower child
                    ax1.plot([i, i+1], [j, j], 'k-', alpha=0.3)
        
        ax1.set_title('Asset Price Tree')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Node')
        ax1.grid(True)
        
        # Plot option tree
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                # Plot node
                ax2.scatter(i, j, color='green')
                # Add option value label
                ax2.annotate(f"{option_tree[j, i]:.2f}", (i, j), 
                           textcoords="offset points", xytext=(0, 5), 
                           ha='center')
                
                # Draw lines to children if not at the last step
                if i < self.n_steps:
                    # Line to upper child
                    ax2.plot([i, i+1], [j, j+1], 'k-', alpha=0.3)
                    # Line to lower child
                    ax2.plot([i, i+1], [j, j], 'k-', alpha=0.3)
        
        ax2.set_title(f'{self.option_type.capitalize()} Option Value Tree')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Node')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def compare_with_black_scholes(self):
        """Compare binomial model price with Black-Scholes price."""
        # Black-Scholes formula for call option
        def black_scholes_call(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        # Black-Scholes formula for put option
        def black_scholes_put(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Calculate binomial price
        binomial_price = self.price_option()
        
        # Calculate Black-Scholes price
        if self.option_type == 'call':
            bs_price = black_scholes_call(self.S0, self.K, self.T, self.r, self.sigma)
        else:
            bs_price = black_scholes_put(self.S0, self.K, self.T, self.r, self.sigma)
        
        return {
            'model': 'Binomial Model' if self.option_type == 'call' else 'Binomial Model (Put)',
            'price': binomial_price,
            'black_scholes': bs_price,
            'difference': binomial_price - bs_price,
            'rel_difference': (binomial_price - bs_price) / bs_price * 100 if bs_price != 0 else float('inf')
        }
    
    def convergence_analysis(self, max_steps=100):
        """Analyze how the binomial price converges to Black-Scholes as steps increase."""
        steps = list(range(5, max_steps + 1, 5))
        binomial_prices = []
        
        # Calculate Black-Scholes price for reference
        def black_scholes_call(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        def black_scholes_put(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        if self.option_type == 'call':
            bs_price = black_scholes_call(self.S0, self.K, self.T, self.r, self.sigma)
        else:
            bs_price = black_scholes_put(self.S0, self.K, self.T, self.r, self.sigma)
        
        # Calculate binomial prices for different numbers of steps
        for n in steps:
            model = BinomialModel(self.S0, self.K, self.T, self.r, self.sigma, n, self.option_type)
            binomial_prices.append(model.price_option())
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.plot(steps, binomial_prices, 'bo-', label='Binomial Price')
        plt.axhline(y=bs_price, color='r', linestyle='-', label='Black-Scholes Price')
        plt.title(f'Convergence of Binomial Model to Black-Scholes ({self.option_type.capitalize()} Option)')
        plt.xlabel('Number of Steps')
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()

# Example usage
if __name__ == "__main__":
    # Parameters
    S0 = 100.0    # Initial stock price
    K = 100.0     # Strike price
    T = 1.0       # Time to expiration in years
    r = 0.05      # Risk-free rate
    sigma = 0.20  # Volatility
    n_steps = 5   # Number of time steps
    
    # Create and use the model
    call_model = BinomialModel(S0, K, T, r, sigma, n_steps, 'call')
    call_price = call_model.price_option()
    print(f"Call option price: ${call_price:.4f}")
    
    # Compare with Black-Scholes
    comparison = call_model.compare_with_black_scholes()
    print(f"Black-Scholes price: ${comparison['black_scholes']:.4f}")
    print(f"Difference: ${comparison['difference']:.4f} ({comparison['rel_difference']:.2f}%)")
    
    # Plot trees
    call_model.plot_trees()
    plt.show()
    
    # Analyze convergence
    call_model.convergence_analysis()
    plt.show()
    
    # Create put option model
    put_model = BinomialModel(S0, K, T, r, sigma, n_steps, 'put')
    put_price = put_model.price_option()
    print(f"Put option price: ${put_price:.4f}")
