import random

class Portfolio():
    def __init__(self):
        self.balance = 0
        self.history = []
        self.stockdict = {}
        self.mystock = {}
        self.mutualfunddict = []
        self.mymutualfund = {}


    def addCash(self, amount_added): # adds amount_added to the balance
        self.balance += amount_added
        print("You added $%d cash to your portfolio. Your current balance is: $%d" % (amount_added, self.balance)) # prints the message for user
        self.history.append("You added $%d cash to your portfolio." % amount_added) # adds this transaction to history list

    def withdrawCash(self, amount_withdrawn): # removes amount_withdrawn from the balance
        if self.balance >= amount_withdrawn: # checks whether balance is enough or not
            self.balance -= amount_withdrawn # if balance is enough, removes amount_withdrawn from the balance
            print("You withdraw $%d cash from your portfolio. Your current balance is: $%d" % (amount_withdrawn, self.balance)) # prints the message for user
            self.history.append("You withdraw $%d cash from your portfolio." % amount_withdrawn) # adds this transaction to history list
        else:
            print("There is no enough cash in your portfolio for this withdraw operation") # if balance is not enough, prints the error message for user

    def createStock(self, price, symbol): # creates stock dictinary
        if symbol in self.stockdict: # checks whether stock is already created or not
            self.stockdict[symbol].add(price) # if stock is already created, it updates price
        else:
            self.stockdict[symbol] = price # if stock is not already created, it creates stock

    def buyStock(self, share, symbol):
        if self.balance >= share * self.stockdict.get(symbol): # checks whether balance is enough or not
            if symbol in self.mystock: # checks customer already has stock or not
                self.mystock[symbol].add(share) # if customer already has this stock, it updates share
            else:
                self.mystock[symbol] = share # if customer don't have this stock, it adds stock to the portfolio
            print("You buy %d shares of stock %s" % (share, symbol)) # prints the message for user
            self.history.append("You buy %d shares of stock %s" % (share, symbol)) # adds this transaction to history list
            portfolio.withdrawCash(share * self.stockdict.get(symbol)) # remove cash from balance to buy stock
        else:
            print("There is no enough cash in your portfolio for this operation") # if balance is not enough, prints the error message for user

    def sellStock(self, share, symbol):
        if self.mystock[symbol] >=share: # it checks whether customer has enough share or not
            self.mystock[symbol] -= share # if customer has enough share, it removes the share from portfolio
            print("You sell %d shares of stock %s" % (share, symbol)) # prints the message for user
            self.history.append("You sell %d shares of stock %s" % (share, symbol)) # adds this transaction to history list
            portfolio.addCash(share * self.stockdict.get(symbol) * random.uniform(0.5,1.5)) # add cash to balance because of selling stocks
        else:
            print("There is no enough share in your portfolio for this operation") #if customer don't have enough share, it prints error message

    def createMutualFund(self, symbol): # creates mutual fund dictinary
        if symbol not in self.mutualfunddict: # checks whether mutual fund is already created or not
            self.mutualfunddict.append(symbol) # # if mutual fund is not already created, it adds mutual fund to the list
        else:
            print("Fund %s is already created" % symbol) # if mutual fund is already created, it prints error message


    def buyMutualFund(self, share, symbol):
        if self.balance >= share * 1: # checks whether balance is enough or not
            if symbol in self.mymutualfund: # checks customer already has mutual fund or not
                self.mymutualfund[symbol].add(share) # if customer already has this mutual fund, it updates share
            else:
                self.mymutualfund[symbol] = share # if customer don't have this mutual fund, it adds mutual fund to the portfolio
            print("You buy %d shares of fund %s" % (share, symbol)) # prints the message for user
            self.history.append("You buy %d shares of fund %s" % (share, symbol)) # adds this transaction to history list
            portfolio.withdrawCash(share * 1) # remove cash from balance to buy mutual fund
        else:
            print("There is no enough cash in your portfolio for this operation") # if balance is not enough, prints the error message for user


    def sellMutualFund(self, share, symbol):
        if self.mymutualfund[symbol] >=share: # it checks whether customer has enough share or not
            self.mymutualfund[symbol] -= share # if customer has enough share, it removes the share from portfolio
            print("You sell %d shares of mutual fund %s" % (share, symbol)) # prints the message for user
            self.history.append("You sell %d shares of mutual fund %s" % (share, symbol)) # adds this transaction to history list
            portfolio.addCash(share * 1 * random.uniform(0.9,1.2)) # add cash to balance because of selling mutual funds
        else:
            print("There is no enough share in your portfolio for this operation") #if customer don't have enouh share, it prints error message

portfolio = Portfolio()
portfolio.addCash(100)
portfolio.withdrawCash(5)
portfolio.withdrawCash(10)
portfolio.createStock(20, "HFH")
#s = Stock()
#s.addStock(20, "HFH")
#print(s.stockdict)
portfolio.buyStock(3, "HFH")
portfolio.sellStock(1, "HFH")
#mf1 = MutualFund("BRT")
#mf2 = MutualFund("GHT")
#print(portfolio.stockdict)
#print(portfolio.stockdict["HFH"])
#print(portfolio.mystock)
portfolio.createMutualFund("BRT")
portfolio.createMutualFund("GHT")
portfolio.createMutualFund("GHT")
portfolio.buyMutualFund(12, "BRT")
portfolio.sellMutualFund(3, "BRT")
print("Your portfolio has $%d Cash and" % portfolio.balance, portfolio.mystock, "as stocks and", portfolio.mymutualfund, "as mutual funds")
print("Your trancastion history:", portfolio.history)
