import wbdata
import datetime
import linearRegression

data_date = datetime.datetime(2019, 1, 1), datetime.datetime(2019, 12, 1)
countries = [i['id'] for i in wbdata.get_country(incomelevel='HIC')]
indicators = {"SH.STA.SUIC.P5": "Suicide mortality rate", "NY.GDP.PCAP.PP.KD": "gdppc"}
df = wbdata.get_dataframe(indicators, country=countries, data_date=data_date)
#print(df)
#print(df.iloc[:,2:])
df.to_csv('/Users/gokhanmurali/OneDrive - Koc Universitesi/Data Science Master/Courses/1. Dönem/CSSM 502/HW2/worldbank.csv',index=False)

linearRegression.linearRegression('worldbank.csv')
