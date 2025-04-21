#!/usr/bin/python3

import json
from collections import OrderedDict
import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pypfopt.expected_returns import capm_return, returns_from_prices
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt, risk_models, EfficientSemivariance, EfficientCVaR, EfficientCDaR


def get_user_input():
    """
    investment_amount: total value of the portfolio
    year_contributions: how much the user is investing this year
    year_salary: how much the user's salary is this year
    risk_tolerance: scale from 1-10 (1 = low tolerance, 10 = high tolerance)
    age: how old the user is
    target_retirement_year: what year the user expects to retire (0 if retired)
    management_comfort_level = 0 (1 = I don't want to manually allocate my funds; 2 = I am open to manually reallocating my funds; 3 = I can comfortably reallocate my funds)
    brokerage: if the user is using Fidelity or Vanguard as their brokerage

    NOTE ---------------------------
    ask desired amount by retirement?
    """

    filepath = "investments.json"
    
    # try:
    with open(filepath, 'r') as f:
        investment = json.load(f)
    print(investment)

    investment = investment[0]

    investment_amount = int(investment["totalInvestment"])
    year_contributions = int(investment["yearlyInvestment"])
    year_salary = int(investment["annualSalary"])
    risk_tolerance = int(investment["riskTolerance"])
    age = int(investment["age"])
    target_retirement_year = int(investment["retirementYear"])
    management_comfort_level = int(investment["portfolioManagement"])
    brokerage = investment["investmentPlatform"]


    current_year = datetime.datetime.today().year

    # If the user is investing too much of their salary (>20%), warn them that they may be investing too much
    # If not currently employed, set percentage of income to 1 to avoid divide by zero error

    percentage_of_income = year_contributions / max(year_salary, 1)

    years_until_retirement = max((target_retirement_year - current_year), 0)

    return investment_amount, risk_tolerance, age, target_retirement_year, management_comfort_level, brokerage, percentage_of_income, years_until_retirement


def get_historical_data(brokerage):
    """
    NOTE: Change to use database when available

    Fidelity-------------------------------------

    Mutual Funds:
    FZROX: Fidelity ZERO Total Market Index Fund
    FZILX: Fidelity ZERO International Index Fund
    FEMKX: Fidelity Emerging Markets Fund

    FNILX: Fidellity ZERO Large Cap Index Fund
    FMSDX: Fideltiy Mid Cap Index Fund
    FSSNX: Fidelity Small Cap Index Fund

    Bonds:
    FXNAX: Fidelity® US Bond Index Fund
    FSHBX: Fidelity Short-Term Bond Fund
    FUAMX: Fidelity Intermediate Treasury Bond Index Fund
    FNBGX: Fideltiy Long-Term Treasury Bond Index Fund

    Target Retirement Funds:
    fidelity_target_retirement_funds = {
    'FQIFX' : '2025',
    'FXIFX' : '2030',
    'FIHFX' : '2035',
    'FBIFX' : '2040',
    'FIOFX' : '2045',
    'FIPFX' : '2050',
    'FDEWX' : '2055',
    'FDKLX' : '2060',
    'FFINX' : '2065',
    'FRBVX' : '2070',
    }

    Vanguard-------------------------------------

    Consider:
    VFIAX
    VTSAX

    Mutual Funds:
    VTI: Vanguard Total Stock Market ETF
    VXUS: Vanguard Total International Stock Index Fund ETF
    VWO: Vanguard FTSE Emerging Markets ETF

    VV: Vanguard Large Cap ETF
    VO: Vanguard Mid Cap ETF
    VB: Vanguard Small Cap ETF

    Bonds:
    BND: Vanguard Total Bond Market ETF
    BSV: Vanguard Short-Term Bond ETF
    VTIP: Vanguard Short-Term Inflation-Protected Securities ETF
    VGIT: Vanguard Intermidiate-Term Treasury ETF
    VGLT: Vanguard Long-Term Treasury ETF

    Target Retirement Funds (0.08% expense ratio, $1,000 minimum investment for all):
    vanguard_target_retirement_funds = {
    'VTTVX' : '2025' 
    'VTHRX' : '2030', 
    'VTTHX' : '2035',
    'VFORX' : '2040',
    'VTIVX' : '2045',
    'VFIFX' : '2050',
    'VFFVX' : '2055',
    'VTTSX' : '2060',
    'VLXVX' : '2065',
    'VSVNX' : '2070',
    }

    Benchmark Indices------------------------------------
    GSPC: S&P 500
    DJI: Dow Jones
    IXIC: NASDAQ
    """

    df = pd.read_csv("final_sample_data.csv", index_col='Date', parse_dates=True)

    # Choose fund options based on the user's brokerage
    if brokerage == 'Vanguard': 
        adj_closing_prices = df.iloc[:, 13:]
    else: 
        adj_closing_prices = df.iloc[:, 3:13]

    # Load benchmark index data
    benchmark_index_data = df.iloc[:, 0:3]

    return adj_closing_prices, benchmark_index_data


def get_allocation_strategy(risk_tolerance, management_comfort_level, percentage_of_income, years_until_retirement):
    """
    Use User input to pick the allocation strategy that best matches the risk the user should take on
    """

    # If the user is investing too much of their salary (>20%), warn them that they may be investing too much
    if percentage_of_income > 0.20:
        print("Warning: you may be investing too much of your income. The suggested range is 10% to 20%.")
        print("Your Portfolio will have reduced risk due to this.")

        # Reduce risk because percentage of income is too high
        risk_tolerance = (1-percentage_of_income) * risk_tolerance

    # Suggest target date fund if user does not want to manage their fund
    if management_comfort_level == 1:
        return 'target date fund'
    
    # Go to retirement allocation strategies if user has already retired
    if years_until_retirement == 0:
        return 'retired'
    
    # Slightly reduce risk if management_comfort_level is 2
    # No risk adjustment if management_comfort_level is 3
    if management_comfort_level == 2:
        risk_tolerance = risk_tolerance * 0.9

    # Modern Portfolio Theory strategies assigned from least to most risk
    if risk_tolerance <= 2:
        reccomendation = 'HRP'
    elif risk_tolerance > 2 and risk_tolerance <= 3.6:
        reccomendation = 'MVO'
    elif risk_tolerance > 3.6 and risk_tolerance  <= 5.2:
        reccomendation = 'Efficient Semivariance'
    elif risk_tolerance > 5.2 and risk_tolerance <= 6.8:
        reccomendation = 'mCVAR'
    elif risk_tolerance > 6.8 and risk_tolerance <= 8.4:
        reccomendation = 'Efficient CDaR'
    elif risk_tolerance > 8.4:
        reccomendation = 'Efficient CVaR'

    return reccomendation

    
def get_target_date_fund(target_retirement_year, brokerage, investment_amount, age, latest_prices):

    fidelity_target_retirement_funds = {
        year: fund 
        for fund, years in {
            'FQIFX': range(2025, 2029),
            'FXIFX': range(2030, 2034),
            'FIHFX': range(2035, 2039),
            'FBIFX': range(2040, 2044),
            'FIOFX': range(2045, 2049),
            'FIPFX': range(2050, 2054),
            'FDEWX': range(2055, 2059),
            'FDKLX': range(2060, 2064),
            'FFINX': range(2065, 2069),
            'FRBVX': range(2070, 2075)
        }.items()
        for year in years
    }

    vanguard_target_retirement_funds = {
        year: fund 
        for fund, years in {
            'VTTHX': range(2020, 2025),
            'VTHRX': range(2030, 2034),
            'VTTVX': range(2035, 2039),
            'VTWNX': range(2040, 2044),
            'VTINX': range(2045, 2049),
            'VFIFX': range(2050, 2054),
            'VFFVX': range(2055, 2059),
            'VTTSX': range(2060, 2064),
            'VLXVX': range(2065, 2069),
            'VSVNX': range(2070, 2075)
        }.items()
        for year in years
    }

    print("To visualize how your fund will change over time, view https://retirementplans.vanguard.com/VGApp/pe/pubeducation/investing/LTgoals/TargetRetirementFunds.jsf")

    # Select the appropriate mapping
    year_to_fund = vanguard_target_retirement_funds if brokerage == 'Vanguard' else fidelity_target_retirement_funds
    index = year_to_fund.get(target_retirement_year)

    if index:
        print(f"At the target retirement year of {target_retirement_year}, your suggested index is {index} in {brokerage}.")
    else:
        print(f"No fund found for target retirement year {target_retirement_year}. You can still follow the estimated asset allocation.")

    weights = get_estimated_asset_allocation(age, brokerage, investment_amount, latest_prices)
    return weights


def get_retired_fund(investment_amount, brokerage, age, latest_prices):
    """
    According to Charles Schwab (https://www.schwab.com/retirement-portfolio):
    * 60-69: moderate portfolio (60% stock, 35% bonds, 5% cash/cash investments)
    * 70–79: moderately conservative (40% stock, 50% bonds, 10% cash/cash investments)
    * 80 and above, conservative (20% stock, 50% bonds, 30% cash/cash investments).

    Currently using:
    Fidelity Managed Retirement Funds: https://www.fidelity.com/mutual-funds/mutual-fund-spotlights/managed-retirement-funds
    Vanguard Target Retirement Funds: https://investor.vanguard.com/investment-products/mutual-funds/target-retirement-funds
    """

    fidelity_year_to_fund = {
        year: fund 
        for fund, years in {
            'FBIFX': range(1968, 1973),
            'FMRTX': range(1963, 1968),
            'FMRAX': range(1958, 1963),
            'FIXRX': range(1953, 1958),
            'FIRVX': range(1948, 1953),
            'FIRSX': range(1943, 1948),
            'FIRQX': range(1938, 1943),
            'FIRMX': range(1920, 1938)
        }.items()
        for year in years
    }

    vanguard_year_to_fund = {
        year: fund 
        for fund, years in {
            'VTTHX': range(1968, 1973),
            'VTHRX': range(1963, 1968),
            'VTTVX': range(1958, 1963),
            'VTWNX': range(1953, 1958),
            'VTINX': range(1920, 1953)
        }.items()
        for year in years
    }

    current_year = datetime.datetime.now().year
    birth_year = current_year - age

    # Select the appropriate mapping
    year_to_fund = vanguard_year_to_fund if brokerage == 'Vanguard' else fidelity_year_to_fund
    index = year_to_fund.get(birth_year)

    if index:
        print(f"At your age of {age} (born in {birth_year}), your suggested index is {index} in {brokerage}.")
    else:
        print(f"No fund found for birth year {birth_year}. You can still follow the estimated asset allocation.")

    weights = get_estimated_asset_allocation(age, brokerage, investment_amount, latest_prices)
    return weights


# Display asset allocation in a pie chart and print exact amounts
def display_asset_allocation(latest_prices, allocation, leftover, allocation_type):
    # Calculate dollar amounts
    dollar_allocation = {ticker: shares * latest_prices[ticker] 
                        for ticker, shares in allocation.items()}
    
    # Show discrete allocation
    print("\nWe suggest buying:")
    for ticker, shares in allocation.items():
        amount = dollar_allocation[ticker]
        print(f"{shares} shares in {ticker}, worth ${amount:,.2f} ")

    # Show remaining cash
    print(f"\nFunds remaining: ${leftover:,.2f}")

    # Create pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(dollar_allocation.values(), 
            labels=dollar_allocation.keys(), 
            autopct='%1.1f%%', 
            startangle=140)
    plt.axis('equal')
    plt.title(f"Allocation from {allocation_type}")
    plt.show()

    return


def HRP(mutual_fund_data, investment_amount, latest_prices):
    mu = returns_from_prices(mutual_fund_data)
    S = CovarianceShrinkage(mutual_fund_data).ledoit_wolf() # Covariance matrix

    # Run optimization algorithm to get weights
    hrp = HRPOpt(mu, S)
    hrp.optimize()
    hrp_weights = hrp.clean_weights()

    hrp.portfolio_performance(verbose=True)
    #print("\nHierarchial risk parity weights:", hrp_weights)

    # Get exact allocation values
    da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = da_hrp.greedy_portfolio()

    display_asset_allocation(latest_prices, allocation, leftover, "Hierarchial Risk Parity (HRP) allocation:")

    return hrp_weights
    

def MVO(mutual_fund_data, investment_amount, latest_prices):
    mu = capm_return(mutual_fund_data)  # Calculated returns
    S = CovarianceShrinkage(mutual_fund_data).ledoit_wolf() # Covariance matrix

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()

    cleaned_weights = ef.clean_weights()

    #print("Mean variance optimization weights:", dict(cleaned_weights))
    ef.portfolio_performance(verbose=True)

    # Get exact allocation values
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = da.greedy_portfolio()

    display_asset_allocation(latest_prices, allocation, leftover, 'Mean Variance Optimization (MVO) allocation:')

    return cleaned_weights


def efficient_semivariance(mutual_fund_data, investment_amount, latest_prices):
    
    mu = capm_return(mutual_fund_data)
    historical_returns = returns_from_prices(mutual_fund_data)

    es = EfficientSemivariance(mu, historical_returns)
    es.efficient_return(0.10) # Looking for a portfolio that minimizes the semivariance for a target annual return of 10%

    weights = es.clean_weights()
    #print("Weights:", weights)
    es.portfolio_performance(verbose=True)

    # Discrete allocation
    da_sv = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)

    allocation, leftover = da_sv.greedy_portfolio()

    display_asset_allocation(latest_prices, allocation, leftover, "Efficient Semivariance allocation:")

    return weights


def mCVAR(mutual_fund_data, investment_amount, latest_prices):

    mu = capm_return(mutual_fund_data) 
    S = mutual_fund_data.cov()
    ef_cvar = EfficientCVaR(mu, S)
    cvar_weights = ef_cvar.min_cvar()

    cleaned_weights = ef_cvar.clean_weights()
    #print("Weights:", dict(cleaned_weights))
    ef_cvar.portfolio_performance(verbose=True)

    # Discrete allocation
    da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = da_cvar.greedy_portfolio()

    display_asset_allocation(latest_prices, allocation, leftover, "Monte Carlo Value at Risk (mCVAR) allocation:")

    return cleaned_weights


def efficient_cdar(mutual_fund_data, investment_amount, latest_prices):
    
    mu = capm_return(mutual_fund_data)
    historical_returns = returns_from_prices(mutual_fund_data)

    es = EfficientCDaR(mu, historical_returns)
    es.efficient_return(0.10) # Looking for a portfolio that minimizes the semivariance for a target annual return of 10%

    weights = es.clean_weights()
    #print("Weights:", weights)
    es.portfolio_performance(verbose=True)

    # Discrete allocation
    da_cdar = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = da_cdar.greedy_portfolio()

    display_asset_allocation(latest_prices, allocation, leftover, "Efficeint Conditional Drawdown at Risk (CDaR) allocation:")

    return weights


def efficient_cvar(mutual_fund_data, investment_amount, latest_prices):

    mu = capm_return(mutual_fund_data)
    historical_returns = returns_from_prices(mutual_fund_data)

    es = EfficientCVaR(mu, historical_returns)
    es.efficient_return(0.10) # Looking for a portfolio that minimizes the semivariance for a target annual return of 10%

    # We can use the same helper methods as before
    weights = es.clean_weights()
    #print("Weights:", weights)
    es.portfolio_performance(verbose=True)

    # Discrete allocation
    da_cvar = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = da_cvar.greedy_portfolio()

    display_asset_allocation(latest_prices, allocation, leftover, "Efficeint Conditional Value at Risk (CVaR) allocation:")

    return weights


# Get the correct index for each asset type in for the right brokerage
# Converts allocation percentages into normalized weights between 0-1
# Used in get_estimated_asset_allocation()
def build_estimated_portfolio(allocation, brokerage, investment_amount, latest_prices):
    # Normalize allocations
    total = sum(allocation)
    normalized_allocation = [a/total for a in allocation]
    
    # Build weights dictionary
    weights = OrderedDict()
    if brokerage == "Vanguard":
        tickers = ["VTI", "VXUS", "BND", "VTIP"]
    else:
        tickers = ["FZROX", "FZILX", "FXNAX", "FSHBX"]
    
    for i, ticker in enumerate(tickers):
        weights[ticker] = normalized_allocation[i]
    
    # Calculate discrete allocation if prices are provided
    allocation_dict = OrderedDict()
    leftover = 0
    
    for ticker, weight in weights.items():
        dollar_amount = weight * investment_amount
        shares = int(dollar_amount / latest_prices[ticker])
        allocation_dict[ticker] = shares
        leftover += dollar_amount - (shares * latest_prices[ticker])

    display_asset_allocation(latest_prices, allocation_dict, leftover, "Estimated retirement portfolio allocation: ")

    return weights

# Getting estimated asset allocation is needed for users who were suggested a target date fund or retired fund strategy
def get_estimated_asset_allocation(age, brokerage, investment_amount, latest_prices):
    """
    Used post-retirement data from Fidelity and target date retirement funds from Vanguard.
    Found that asset allocation did not change between 20-40 years old (target date retirement 2070-2050)
    Got accurate asset allocation estimates from a trendline for the rest of the ages (40+)
    """
    # Asset order: US-stocks, bonds, Non-US stocks, short term debt 

    # If the user is under 40, the asset allocation does not change
    if age < 40:
        asset_allocation = [54, 36, 10, 0]

    # If the user is 40 or older, use linear equations to estimate asset allocation
    else: 
        US_equity = max(0, (-0.976 * age) + 93.6)
        non_US_equity = max(0, (-0.63 * age) + 60.9)
        bonds = max(0, (1.37 * age) - 42.8)

        # No short term debt until over 60 years old
        short_term_debt = max(0, (0.37 * age) - 21.5) if age >0 else 0
        
        # Put all percentages together
        asset_allocation = [US_equity, non_US_equity, bonds, short_term_debt]

    # Get the designated index for each asset
    asset_allocation = build_estimated_portfolio(asset_allocation, brokerage, investment_amount, latest_prices)

    return asset_allocation


# Monte Carlo method to simulate potential forecasts of the portfolio and benchmark indices
# Correlating benchmarks with portfolio for more realistic results (correlated, move together in reality)
def portfolio_monte_carlo_simulation(mutual_fund_data, benchmark_index_data, weights, investment_amount):
    # Set simulation parameters
    simulations = 100
    days = 360 * 5

    # Verify all weight keys exist in mutual fund data
    missing_funds = [f for f in weights.keys() if f not in mutual_fund_data.columns]
    if missing_funds:
        raise ValueError(f"Weight keys not found in mutual fund data: {missing_funds}")
    
    # Filter mutual fund data to only include weighted funds
    filtered_mutual_fund_data = mutual_fund_data[list(weights.keys())]

    # Combine all indices (portfolio + benchmarks) for correlated simulation
    all_returns = pd.concat([
        filtered_mutual_fund_data.pct_change(),
        benchmark_index_data.pct_change()
    ], axis=1).dropna()
    
    # Get combined statistics
    mean_returns = all_returns.mean().values
    cov_matrix = all_returns.cov().values
    n_portfolio = len(weights)
    n_benchmarks = len(benchmark_index_data.columns)

    # Initialize results
    portfolio_sims = np.zeros((days, simulations))
    benchmarks_sims = np.zeros((days, simulations, n_benchmarks))
    weights_array = np.array(list(weights.values()))

    for sim in range(simulations):
        # Generate correlated random numbers for all assets
        Z = np.random.normal(size=(days, n_portfolio + n_benchmarks))
        L = np.linalg.cholesky(cov_matrix)
        correlated_returns = mean_returns + (Z @ L.T)
        
        # Split into portfolio and benchmark returns
        portfolio_daily_returns = correlated_returns[:, :n_portfolio]
        benchmark_daily_returns = correlated_returns[:, n_portfolio:]
        
        # Portfolio simulation
        weighted_returns = portfolio_daily_returns @ weights_array
        portfolio_sims[:, sim] = investment_amount * np.cumprod(1 + weighted_returns)
        
        # Benchmarks simulation
        benchmarks_sims[:, sim, :] = investment_amount * np.cumprod(1 + benchmark_daily_returns, axis=0)

    # Portfolio plot
    plt.figure(figsize=(14, 6))
    plt.plot(portfolio_sims)
    plt.title(f"Portfolio Simulations (Initial Investment: ${investment_amount:,.2f})", fontsize=14)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.xlabel("Trading Days", fontsize=12)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.show()
    
    # Benchmarks plot
    plt.figure(figsize=(14, 6))
    for i in range(n_benchmarks):
        plt.plot(benchmarks_sims[:, :, i], label=benchmark_index_data.columns[i])
    plt.title(f"Benchmark Simulations (Initial Investment: ${investment_amount:,.2f})", fontsize=14)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.xlabel("Trading Days", fontsize=12)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.show()

    return portfolio_sims, benchmarks_sims


# Visualize the 5th, 25th, 50th, 75th, and 95th percentile outcomes to better understand/visualize risk and return of the portfolio, anc relative to the market
def show_percentiles(portfolio_sims, benchmark_sims, benchmark_index_data, investment_amount):
    # Portfolio Percentiles (Standalone Figure)
    plt.figure(figsize=(16, 8))
    portfolio_percentiles = np.percentile(portfolio_sims, [95, 75, 50, 25, 5], axis=1)
    lines = plt.plot(portfolio_percentiles.T, linewidth=2)
    plt.title(f"Portfolio Value Percentiles (Initial: ${investment_amount:,.0f})", fontsize=14)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.xlabel("Trading Days", fontsize=12)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    colors = ['forestgreen', 'blueviolet', 'steelblue', 'orange', 'red']
    for line, color in zip(lines, colors):
        line.set_color(color)
    
    plt.legend(['95th', '75th', 'Median', '25th', '5th'], fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Benchmark Percentiles
    plt.figure(figsize=(16, 8))

    # Define benchmark-specific line styles
    benchmark_line_styles = [
        {'linestyle': '-'},       
        {'linestyle': '--'},   
        {'linestyle': '-.'}
    ]

    # Define percentile colors
    percentile_colors = {
        '5th': 'red',
        '25th': 'orange',
        'Median': 'steelblue',
        '75th': 'blueviolet',
        '95th': 'forestgreen'
    }

    # Plot order
    percentiles = ['95th', '75th', 'Median', '25th', '5th']
    
    for i in range(benchmark_sims.shape[2]):
        benchmark_percentiles = np.percentile(benchmark_sims[:, :, i], [5, 25, 50, 75, 95], axis=1)
        
        # Plot percentiles
        for j, p in enumerate(reversed(percentiles)):
            original_p = ['5th', '25th', 'Median', '75th', '95th'][j]
            style = {
                'color': percentile_colors[original_p],
                'linewidth': 2 if original_p == 'Median' else 1,
                'alpha': 0.7 if original_p != 'Median' else 1,
                'linestyle': benchmark_line_styles[i]['linestyle']
            }
            plt.plot(benchmark_percentiles[j], 
                    label=f"{benchmark_index_data.columns[i]} {p}",
                    **style)

    plt.title("All Benchmark Percentiles Comparison", fontsize=16)
    plt.ylabel("Portfolio Value ($)", fontsize=14)
    plt.xlabel("Trading Days", fontsize=14)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.grid(True, alpha=0.3)

    # Create custom legend
    legend_elements = []
    
    # Add percentile legend items
    for p in percentiles:
        original_p = {'95th': '95th', '75th': '75th', 'Median': 'Median', 
                     '25th': '25th', '5th': '5th'}[p]
        legend_elements.append(Line2D([0], [0], 
                                color=percentile_colors[original_p],
                                linewidth=2 if original_p == 'Median' else 1,
                                label=p))
    
    # Add benchmark style legend items
    for i, benchmark in enumerate(benchmark_index_data.columns):
        legend_elements.append(Line2D([0], [0], 
                                color='black',
                                linestyle=benchmark_line_styles[i]['linestyle'],
                                label=f"{benchmark} style"))

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

    return


# Displays a plot with percentiles from both the portfolio and benchmarks
def show_combined_percentiles(portfolio_sims, benchmark_sims, benchmark_index_data, investment_amount):
    plt.figure(figsize=(14, 8))
    
    # Calculate portfolio percentiles
    portfolio_percentiles = np.percentile(portfolio_sims, [95, 75, 50, 25, 5], axis=1)
    
    # Plot portfolio percentiles
    plt.plot(portfolio_percentiles.T, 
             label=['Portfolio 95th', 'Portfolio 75th', 'Portfolio Median', 
                    'Portfolio 25th', 'Portfolio 5th'],
             linestyle='-', linewidth=2)
    
    # Calculate and plot benchmark percentiles
    n_benchmarks = benchmark_sims.shape[2]
    colors = plt.cm.tab10(np.linspace(0, 1, n_benchmarks))  # Different colors for each benchmark
    
    for i in range(n_benchmarks):
        benchmark_name = benchmark_index_data.columns[i]
        benchmark_percentiles = np.percentile(benchmark_sims[:, :, i], [50], axis=1)
        
        plt.plot(benchmark_percentiles.T, 
                 label=f'{benchmark_name} Median',
                 color=colors[i], 
                 linestyle='--', 
                 linewidth=2)
    
    plt.title(f"Projected Portfolio vs Benchmark Percentiles (Initial: ${investment_amount:,.0f})", fontsize=14)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.xlabel("Trading Days", fontsize=12)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return


def main():
    # Read user input from frontend 
    # Parse user information
    investment_amount, risk_tolerance, age, target_retirement_year, management_comfort_level, brokerage, percentage_of_income, years_until_retirement = get_user_input()

    # Print user input
    print("Investment amount: ", investment_amount)
    print("Risk tolerance: ", risk_tolerance)
    print("Age: ", age)
    print("Target retirement year: ", target_retirement_year)
    print("Management comfort level: ", management_comfort_level)
    print("Brokerage: ", brokerage)
    print("percentage_of_income: ", percentage_of_income)
    print("Years until retirement: ", years_until_retirement)

    # Choose allocation strategy
    allocation_strategy = get_allocation_strategy(risk_tolerance, management_comfort_level, percentage_of_income, years_until_retirement)
    print(f"Your suggested allocation strategy is {allocation_strategy}!")

    # Collect Historical Price Data 
    mutual_fund_data, benchmark_index_data = get_historical_data(brokerage)
    latest_prices = get_latest_prices(mutual_fund_data)

    # Get portfolio allocation based on suggested allocation strategy
    if allocation_strategy == 'target date fund':
        if (investment_amount < 1000) and (brokerage == "Vanguard"):
            print("DISCLAIMER: Minimum investment for Vanguard retirement funds are $1,000")
        weights = get_target_date_fund(target_retirement_year, brokerage, investment_amount, age, latest_prices)
    elif allocation_strategy == 'retired':
        if (investment_amount < 1000) and (brokerage == "Vanguard"):
            print("DISCLAIMER: Minimum investment for Vanguard retirement funds are $1,000")
        weights = get_retired_fund(investment_amount, brokerage, age, latest_prices)
    elif allocation_strategy == 'HRP':
        weights = HRP(mutual_fund_data, investment_amount, latest_prices)
    elif allocation_strategy == 'MVO':
        weights = MVO(mutual_fund_data, investment_amount, latest_prices)
    elif allocation_strategy == 'Efficient Semivariance':
        weights = efficient_semivariance(mutual_fund_data,investment_amount, latest_prices)
    elif allocation_strategy == 'mCVAR':
        weights = mCVAR(mutual_fund_data, investment_amount, latest_prices)
    elif allocation_strategy == 'Efficient CDaR':
        weights = efficient_cdar(mutual_fund_data,investment_amount, latest_prices)
    else:
        weights = efficient_cvar(mutual_fund_data,investment_amount, latest_prices)

    portfolio_sims, benchmark_sims = portfolio_monte_carlo_simulation(mutual_fund_data, benchmark_index_data, weights, investment_amount)
    show_percentiles(portfolio_sims, benchmark_sims, benchmark_index_data, investment_amount)
    show_combined_percentiles(portfolio_sims, benchmark_sims, benchmark_index_data, investment_amount)
    return


if __name__ == "__main__":
    main()

"""
TODO: 
User input to see how far into the future to project
Include recurring annual investments in simulations?
"""
