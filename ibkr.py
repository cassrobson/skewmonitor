from ibapi.client import *
from ibapi.wrapper import *
from ibapi.contract import Contract
import threading
import time
from decimal import Decimal

class TradeApp(EWrapper, EClient): 
    def __init__(self): 
        EClient.__init__(self, self) 

    def tickGeneric(self, reqId: TickerId, tickType: TickType, value: float):
        print("TickGeneric. TickerId:", reqId, "TickType:", tickType, "Value:", value)

    def tickPrice(self, reqId: TickerId, tickType: TickType, price: float, attrib: TickAttrib):
        print(reqId, tickType, price, attrib)

    def tickSize(self, reqId: TickerId, tickType: TickType, size: Decimal):
        print("TickSize. TickerId:", reqId, "TickType:", tickType, "Size: ", size)

    def tickString(self, reqId: TickerId, tickType: TickType, value: str):
        print("TickString. TickerId:", reqId, "Type:", tickType, "Value:", value)

    def tickReqParams(self, tickerId:int, minTick:float, bboExchange:str, snapshotPermissions:int):
        print("TickReqParams. TickerId:", tickerId, "MinTick:", minTick, "BboExchange:", bboExchange, "SnapshotPermissions:",snapshotPermissions)

    def tickOptionComputation(self, reqId: TickerId, tickType: TickType, tickAttrib: int, impliedVol: float, delta: float, optPrice: float, pvDividend: float, gamma: float, vega: float, theta: float, undPrice: float):
        print("TickOptionComputation. TickerId:", reqId, "TickType:", tickType, "TickAttrib:", tickAttrib, "ImpliedVolatility:", impliedVol, "Delta:", delta, "OptionPrice:", optPrice, "pvDividend:", pvDividend, "Gamma: ", gamma, "Vega:", vega, "Theta:", theta, "UnderlyingPrice:", undPrice)

def websocket_con():
    app.run()
    
app = TradeApp()      
app.connect("127.0.0.1", 7497, clientId=1)

con_thread = threading.Thread(target=websocket_con, daemon=True)
con_thread.start()

time.sleep(1) 

contract = Contract()
contract.symbol = "AAPL"
contract.secType = "OPT"
contract.exchange = "SMART"
contract.currency = "USD"
contract.lastTradeDateOrContractMonth = 20250214
contract.strike = 200
contract.right = "C"
contract.multiplier = 100

app.reqMarketDataType(1)
app.reqMktData(102, contract, "", False, False, [])