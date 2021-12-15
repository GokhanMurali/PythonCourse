import wbdata
import datetime
import linearRegression

data_date = datetime.datetime(2019, 1, 1), datetime.datetime(2019, 12, 1) #I am investigating 2019 data only
countries = [i['id'] for i in wbdata.get_country(incomelevel='HIC')] #My control variable is income level. I fixed this variable. I choose HIC (High Income) countries 
indicators = {"SH.STA.SUIC.P5": "Suicide mortality rate", "NY.GDP.PCAP.PP.KD": "gdppc"} #I am using these data from World Bank API
df = wbdata.get_dataframe(indicators, country=countries, data_date=data_date)
#print(df)
#print(df.iloc[:,2:])
df.to_csv('/Users/gokhanmurali/OneDrive - Koc Universitesi/Data Science Master/Courses/1. Dönem/CSSM 502/HW2/worldbank.csv',index=False) #I am writing data to csv file

linearRegression.linearRegression('worldbank.csv') #I am calling my regression function
