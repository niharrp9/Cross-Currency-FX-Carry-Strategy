# Cross Currency FX Carry Strategy Based on GBP as Primary Currency

This repository presents an in-depth study and implementation of an FX Carry Strategy. The strategy explores the dynamics of borrowing in a low-interest-rate currency and lending in a higher-yielding one, facilitated through cross-currency swaps. This project encapsulates the synthesis of profit and loss (P&L) of weekly-traded cross-currency fixed-float swaps, leveraging data from overnight index swaps (OIS) and various currency pair swap yield curves.

## Introduction

FX Carry Strategy is a well-known financial tactic among traders and involves capitalizing on the differential in interest rates between two currencies. This project uses the GBP as a base for comparison against currencies like the Egyptian Pound, Hungarian Forint, Costa Rican Colon, and Romanian Leu, implementing a quantitative approach to identify and capture arbitrage opportunities.

## Data

The project employs rates from UK OIS and spot FX rates for USD/GBP, as well as swap yield curves and FX rates of various currencies. The rates were obtained from the Bank of England and the Quandl CUR database, with some interpolation needed due to the sporadic nature of maturity data availability.

## Strategy Details

### Fixed-Float Carry
- **Borrowing Currency**: Assumed OIS+50 basis points rate paid on 4/5 the notional amount, simulating 5x leverage.
- **Lending Currency**: Assumed quarterly coupons at the 5-year swap rate.
- **Position Management**: Active positions were opened weekly, with no positions assumed if the interest rate differential was under 50 basis points.

### Mark to Market
- **Lending Currency**: Calculations involve adjusting bond prices based on the latest swap curves and converting values to USD at prevailing FX rates.
- **Borrowing Currency**: Approximated to have no significant value change due to short durations, with accrued interest computed at the borrowing rate.

## Analysis

The analysis component of this project focuses on the performance evaluation of the strategy. It examines:
- **Correlations**: Assessment of inter-relationships between various currency pairs.
- **Market Risk Factors**: Analysis of how the strategy behaves in response to underlying market risks.

## Usage

Interested parties can clone this repository to replicate the strategy, evaluate its effectiveness, and modify it according to individual market perspectives.

## Contributions

Contributors are welcome to fork this repository, propose changes, and submit pull requests. This project is open for collaboration to refine the strategy and expand its scope to additional currency pairs or different financial contexts.

## Insights and Further Research

The repository also discusses potential enhancements and the impact of extended training and testing periods on the strategy's robustness. It provides a ground for further research into the stability of the regression coefficient (β) and the strategic setting of threshold values (j).

## Contact

For discussions on methodology, performance, and potential collaborations, please reach out through the repository's discussion forum or contact directly via email.

