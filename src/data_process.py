

import portSim


portSim.load(mdb="/media/toni/Dados/Git/data/portSimData/marketDataUS.sqlite")


from portSim.simulation import backtestFunctions as bf
from portSim.simulation.Rebalance import Rebalance

d = {'app': 'SP500',
     'date1': '2022-01-01', 
     'date2': '2022-02-28',
     'rebalanceFrequency': 10,
     'inSample': 600,
     'includedAssets': ['AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL', 'TSLA', 'BRK.B'] 
     # 'excludedAssets': 'SPY'
     }

data, settings = bf.dataLoader(d, returnSettings=1, debug=0)



print(data.benchmarkPrices)
print(data.assetPrices)
print(data.adjustedPrices)
print(data.eligibility)
print(data.rebalances)
print(data.firstOutOfSampleDay)
print(data.rebalances[data.firstOutOfSampleDay:])
print(data.dates[data.firstOutOfSampleDay:])
print(data.assets)
print(data.benchmarks)
print(data.tickers)
print(data.assetReturns)
print(data.cash)
print(settings)
print()


rebalance = Rebalance(data.firstHistoryDay, data.getCurrentPrices(data.firstHistoryDay))